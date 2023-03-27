from turtle import forward
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x
    
def uncertainty(depth_probs, u_thre=10):
    """
    Get the certainty mask whose entropy is lower than the threshold.

    Parameters
    ----------
    depth_probs : estimated depth distribution (b, num_agents, d, h, w)

    Returns
    -------
    uncertainty_mask : uncertainty mask (b, num_agents, h, w)
    """
    entropy = - depth_probs * torch.log(depth_probs + 1e-6)
    entropy = entropy.sum(dim=-3)
    # return entropy
    uncertainty_mask = (entropy < u_thre) * 1.0
    return uncertainty_mask

def get_colla_feats(x, record_len, pairwise_t_matrix):
    # (B,L,L,2,3)
    _, C, H, W = x.shape
    B, L = pairwise_t_matrix.shape[:2]
    split_x = regroup(x, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
        # update each node i
        num_agents = batch_node_features[b].shape[0]
        neighbor_features = []
        for i in range(num_agents):
            # i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            neighbor_features.append(neighbor_feature.unsqueeze(0))
        neighbor_features = torch.cat(neighbor_features, dim=0) # num_agents * num_agents * c * h * w
        out.append(neighbor_features)
    return out

def cost_volume(feats, depth_probs, record_len, pairwise_t_matrix, masked_depth_probs=None, valid_depth_thre=0.1):
    """
    Get the consistency weights across multi agents' view.

    Parameters
    ----------
    feats: feature maps (b, num_agents, d, c, h, w)
    depth_probs : estimated depth distribution (b, num_agents, d, h, w)
    masked_depth_probs: estimated depth distribution filtered by uncertanty mask  (b, num_agents, d, h, w)
    pairwise_t_matrix: transformation matrix (b, num_agents, 3, 3)
    valid_depth_thre: depth threshold, filter out the inconfidence estimations (scalar)

    Returns
    -------
    updated_depth_probs: promoted depth distribution by consistency (b, num_agents, d, h, w)
    """
    # warp feature into ego agent's coordinate system
    val_feats = get_colla_feats(feats, record_len, pairwise_t_matrix)   # [(num_agents, num_agents, C,H,W), ...]
    if masked_depth_probs is None:
        val_probs = get_colla_feats(depth_probs, record_len, pairwise_t_matrix)
    else:
        val_probs = get_colla_feats(masked_depth_probs, record_len, pairwise_t_matrix)

    B = len(record_len)
    cost_v = []
    for b in range(B):
        # get consistency score
        val_feat = val_feats[b] # (num_agents,num_agents, c, h, w)
        val_prob = val_probs[b] # (num_agents,num_agents, h, w)
        num_agents = val_feat.shape[0]
        cost_a = []
        for i in range(num_agents):
            sim_score = (val_feat[i,i:i+1] * val_feat[i,:]).sum(dim=-3)    # (num_agents, h, w)
            binary_mask = (val_prob[i,:] > valid_depth_thre).squeeze(1)   # (num_agents, h, w)
            s = (sim_score * binary_mask).sum(dim=0).unsqueeze(0)  # (1, h, w)
            cost_a.append(s)
        cost_a = torch.cat(cost_a, dim=0).unsqueeze(1) # (num_agents, 1, h, w)
        cost_v.append(cost_a)
        
    cost_v = torch.cat(cost_v, dim=0) # (sum(record_len), 1, h, w)
    return cost_v

class CollaDepthNet(nn.Module):
    def __init__(self, dim=128, downsample_rate=1, discrete_ratio=1):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.discrete_ratio = discrete_ratio
        self.norm = nn.LayerNorm(dim)
        self.depth_net = nn.Sequential(
                                    nn.Conv2d(2, 32,
                                        kernel_size=7, padding=3, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1,
                                        kernel_size=3, padding=1, bias=True))
    
    def get_t_matrix(self, pairwise_t_matrix, H, W, downsample_rate, discrete_ratio):
        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2
        return pairwise_t_matrix
    
    def forward(self, x, record_len, pairwise_t_matrix, depth_probs):
        # Multi-view matching
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).flatten(0,2)
        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0,3,1,2)

        pairwise_t_matrix = self.get_t_matrix(pairwise_t_matrix, H, W, downsample_rate=1, discrete_ratio=self.discrete_ratio)
        u_masks = uncertainty(depth_probs, u_thre=10)
        # masked_depth_probs = u_masks.unsqueeze(1) * depth_probs

        cost_v = cost_volume(x, depth_probs, record_len, pairwise_t_matrix, masked_depth_probs=None, valid_depth_thre=0.1) # (B, 1, H, W)
        
        updated_depth = torch.cat([depth_probs, cost_v], dim=1)
        updated_depth = self.depth_net(updated_depth)

        # updated_depth_left = updated_depth[:, :, :H//2,:].softmax(dim=-2)
        # updated_depth_right = updated_depth[:, :, H//2:,:].softmax(dim=-2)
        # updated_depth_lr = torch.cat([updated_depth_left, updated_depth_right], dim=-2)
        # updated_depth_front = updated_depth[:, :, :, :W//2].softmax(dim=-1)
        # updated_depth_back = updated_depth[:, :, :, W//2:].softmax(dim=-1)
        # updated_depth_fb = torch.cat([updated_depth_front, updated_depth_back], dim=-1)
        # updated_depth = (updated_depth_lr + updated_depth_fb) * 0.5

        updated_depth = updated_depth.sigmoid()
        return updated_depth
