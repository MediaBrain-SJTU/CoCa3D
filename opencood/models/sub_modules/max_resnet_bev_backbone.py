import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock, Bottleneck
from opencood.models.sub_modules.auto_encoder import AutoEncoder
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
DEBUG = False

class MaxResNetBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        
        self.discrete_ratio = model_cfg['voxel_size'][0]
        self.downsample_rate = 1


            
        layer_nums = model_cfg['layer_nums']
        num_filters = model_cfg['num_filters']
        layer_strides = model_cfg['layer_strides']
        upsample_strides = model_cfg['upsample_strides']
        num_upsample_filters = model_cfg['num_upsample_filter']
        self.level_num = len(layer_nums)

        self.resnet = ResNetModified(BasicBlock, 
                                    layer_nums,
                                    layer_strides,
                                    num_filters)



        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            
            fuse_network = MaxFusion()
            self.fuse_modules.append(fuse_network)
            
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[idx], num_upsample_filters[idx],
                        upsample_strides[idx],
                        stride=upsample_strides[idx], bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx],
                                    eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ))


        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        if DEBUG:
            origin_features = torch.clone(spatial_features)

        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        ups = []
        ret_dict = {}
        x = spatial_features

        H, W = x.shape[2:]   #  200, 704
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]

        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2


        features = self.resnet(x)  # tuple of features
        ups = []

        for i in range(self.level_num):
            if DEBUG:
                self.fuse_modules[i].forward_debug(features[i], origin_features, record_len, pairwise_t_matrix)
            else:
                x_fuse = self.fuse_modules[i](features[i], record_len, pairwise_t_matrix)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.level_num:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict




class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The affine transformation matrix from each cav to ego, already normalized
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(x, record_len)

        batch_node_features = split_x

        out = []
        # iterate each batch
        for b in range(B):

            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)
        
        return out
