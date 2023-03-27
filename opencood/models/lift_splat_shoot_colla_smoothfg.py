"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler

Intermediate fusion for camera based collaboration
"""

from numpy import record
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from icecream import ic
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization
from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, Up, CamEncode, CamEncodeGTDepth, BevEncode
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.homography import CollaDepthNet
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os
import seaborn as sns
import numpy as np


def FusionNet(fusion_args):
    if fusion_args['core_method'] == "max":
        from opencood.models.fuse_modules.max_fuse import MaxFusion
        return MaxFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'att':
        from opencood.models.fuse_modules.att_fuse import AttFusion
        return AttFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'v2vnet':
        from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion
        return V2VNetFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'v2xvit':
        from opencood.models.fuse_modules.v2xvit_fuse import V2XViTFusion
        return V2XViTFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'when2comm':
        from opencood.models.fuse_modules.when2com_fuse import When2comFusion
        return When2comFusion(fusion_args['args'])
    else:
        raise("Fusion method not implemented.")

class LiftSplatShootCollaSmoothfg(LiftSplatShoot):
    def __init__(self, args): 
        super(LiftSplatShootCollaSmoothfg, self).__init__(args)

        fusion_args = args['fusion_args']
        self.ms = args['fusion_args']['core_method'].endswith("ms")
        if self.ms:
            self.bevencode = BevEncodeMSFusion(fusion_args)
        else:
            self.fusion = FusionNet(fusion_args)
        self.supervise_single = args['supervise_single']

        self.colla_depth = CollaDepthNet(dim=128, downsample_rate=1, discrete_ratio=fusion_args['args']['voxel_size'][0])

        self.cls_head_2d = nn.Sequential(
                                    nn.Conv2d(self.camC, 32,
                                        kernel_size=1, padding=0, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1,
                                        kernel_size=1, padding=0, bias=True))
        kernel_size = 5
        c_sigma = 0.5
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.init_gaussian_filter(self.gaussian_filter, kernel_size, c_sigma)
        self.gaussian_filter.requires_grad = False
        self.vis_count = 0
    
    def init_gaussian_filter(self, gaussian_filter, k_size=5, sigma=1):
        def gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
        gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        gaussian_filter.bias.data.zero_()
        
    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        fg_mask = x[:,-1:,:,:]

        depth_logit, depth_gt_indices, depth_gt, ori_x, x = self.camencode(x) # 进行图像编码 ori_x: B*N x C x fH x fW, x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)

        # 2d classification confidence
        cls_2d = self.cls_head_2d(ori_x)
        cls_2d = cls_2d.sigmoid()
        down_ratio_h = fg_mask.shape[-2]//cls_2d.shape[-2]
        down_ratio_w = fg_mask.shape[-1]//cls_2d.shape[-1]
        assert down_ratio_h==down_ratio_w
        fg_mask = F.max_pool2d(fg_mask, kernel_size=down_ratio_h, stride=down_ratio_h)

        ori_x = ori_x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        ori_x = ori_x.unsqueeze(3).expand(-1, -1, -1, self.D, -1, -1)
        ori_x = ori_x.permute(0, 1, 3, 4, 5, 2)

        depth_prob = F.softmax(depth_logit, dim=1)

        depth_prob_fg = depth_prob * cls_2d

        depth_prob = depth_prob.view(B, N, 1, self.D, imH//self.downsample, imW//self.downsample)
        depth_prob = depth_prob.permute(0, 1, 3, 4, 5, 2)

        depth_prob_fg = depth_prob_fg.view(B, N, 1, self.D, imH//self.downsample, imW//self.downsample)
        depth_prob_fg = depth_prob_fg.permute(0, 1, 3, 4, 5, 2)

        depth_gt_fg = depth_gt * fg_mask
        depth_gt_fg = depth_gt_fg.view(B, N, 1, self.D, imH//self.downsample, imW//self.downsample)
        depth_gt_fg = depth_gt_fg.permute(0, 1, 3, 4, 5, 2)

        depth_gt = depth_gt.view(B, N, 1, self.D, imH//self.downsample, imW//self.downsample)
        depth_gt = depth_gt.permute(0, 1, 3, 4, 5, 2)

        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)
        return ori_x, x, [depth_logit, depth_gt_indices, fg_mask], (depth_prob, depth_gt), (cls_2d, fg_mask), (depth_prob_fg, depth_gt_fg)
    

    def forward(self, data_dict):
        # if "image_inputs" not in data_dict: # LateFusionDataset
        #     image_inputs_dict = {}
        #     image_inputs_dict['imgs'] = data_dict['imgs']
        #     image_inputs_dict['rots'] = data_dict['rots']
        #     image_inputs_dict['trans'] = data_dict['trans']
        #     image_inputs_dict['intrins'] = data_dict['intrins']
        #     image_inputs_dict['post_rots'] = data_dict['post_rots']
        #     image_inputs_dict['post_trans'] = data_dict['post_trans']

        #     data_dict['image_inputs'] = image_inputs_dict
        #     N = data_dict['imgs'].shape[0] # pretend to be N samples, each sample contains 1 cav 
        #     data_dict['record_len'] = torch.tensor([1]*N, dtype=torch.long).to("cuda") # each sample contains 1 cav. make N to be batchsize
        #     data_dict['pairwise_t_matrix'] = torch.eye(4).expand(N,1,1,4,4).to("cuda")

        # print("record_len:", data_dict['record_len'])

        if self.ms:
            return self._forward_ms(data_dict)
        else:
            return self._forward_ss(data_dict)

    def _forward_ms(self, data_dict):
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']

        # Warp to BEV
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)
        x, _, depth_items, depth_items_global, cls_2d_items, depth_items_global_fg = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240

        depth_probs = self.voxel_pooling(geom, depth_items_global[0])
        depth_gts = self.voxel_pooling(geom, depth_items_global[1].float())
        depth_gts = self.gaussian_filter(depth_gts)
        depth_gts_neg = (depth_gts==0) * 1.0 
        
        # Multi-view matching & Tune depth probs
        updated_depth_probs = self.colla_depth(x, record_len, data_dict['pairwise_t_matrix'], depth_probs)
        x = x * updated_depth_probs

        # 2D objectiveness --> find foreground objects --> supervise 3d depth
        # depth_probs_fg = self.voxel_pooling(geom, depth_items_global_fg[0])
        depth_gts_fg = self.voxel_pooling(geom, depth_items_global_fg[1].float())
        depth_gts_fg = (depth_gts_fg>0) * 1.0 
        depth_gts_valid = torch.max(depth_gts_fg, depth_gts_neg)
        updated_depth_probs_valid = updated_depth_probs * depth_gts_valid

        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x_single, x_fuse = self.bevencode(x, record_len, pairwise_t_matrix)
        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'depth_items': depth_items,
                       'depth_fg_items': (updated_depth_probs_valid, depth_gts_fg, depth_gts_valid)}
        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dm": dm})

        if self.supervise_single:
            psm_single = self.cls_head(x_single)
            rm_single = self.reg_head(x_single)
            output_dict.update({'psm_single': psm_single,
                                'rm_single': rm_single,
                                'cls_2d_single': cls_2d_items})
            if self.use_dir:
                dm_single = self.dir_head(x_single)
                output_dict.update({"dm_single": dm_single})
        
        DEBUG = False # True # 
        if DEBUG:
            inputs = image_inputs_dict['imgs']
            B, N, C, imH, imW = inputs.shape
            images = inputs[:,:,:3]
            from opencood.utils.camera_utils import denormalize_tensor
            for b in range(B):
                for n in range(N):
                    images[b,n] = denormalize_tensor(images[b,n])
            depth_images = inputs[:,:,3]
            fg_mask = cls_2d_items[1].view(B, N, cls_2d_items[0].shape[-2], cls_2d_items[0].shape[-1])
            est_fg_mask = cls_2d_items[0].view(B, N, cls_2d_items[0].shape[-2], cls_2d_items[0].shape[-1])
            est_depth_images = depth_items[0].view(B, N, self.D, depth_items[0].shape[-2], depth_items[0].shape[-1]).max(dim=2)[1]
            gt_depth_images = depth_items[1].view(B, N, depth_items[0].shape[-2], depth_items[0].shape[-1])
            
            est_depth_bev = depth_probs.max(dim=1)[1]
            updated_est_depth_bev = updated_depth_probs.max(dim=1)[1]
            gt_depth_valid_mask = depth_gts_valid.squeeze(1)
            gt_depth_fg_mask = depth_gts_fg.squeeze(1)
            est_obj_bev = psm_single.sigmoid().max(dim=1)[0]

            save_dir = '/GPFS/data/yhu/code_camera/OpenCOOD/vis_result/depth/{}'.format(self.vis_count)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for i in range(B):
                # depth at image coordinate
                fig, axes = plt.subplots(5, 4, figsize=(12,12))
                for j in range(N):
                    axes[0,j].imshow((images[i,j].detach().cpu().numpy()[::-1].transpose(1,2,0) * 255.).astype('uint8')) # (h, w)
                    sns.heatmap(depth_images[i,j].detach().cpu().numpy(), cbar=True, ax=axes[1,j], vmin=0, vmax=self.D)
                    sns.heatmap(est_depth_images[i,j].detach().cpu().numpy(), cbar=True, ax=axes[2,j], vmin=0, vmax=self.D)
                    sns.heatmap(fg_mask[i,j].detach().cpu().numpy(), cbar=True, ax=axes[3,j])
                    sns.heatmap(est_fg_mask[i,j].detach().cpu().numpy(), cbar=True, ax=axes[4,j])

                for ax_i in range(5):
                    for ax_j in range(4):
                        axes[ax_i, ax_j].set_axis_off()
                fig.tight_layout()
                
                plt.savefig(os.path.join(save_dir, 'imagecoord_{}.png'.format(i)))
                plt.close()

                # depth at bev coordinate
                fig, axes = plt.subplots(1, 5, figsize=(10,2))
                
                sns.heatmap(est_depth_bev[i].detach().cpu().numpy(), cbar=True, ax=axes[0], vmin=0, vmax=1) # (h, w)
                sns.heatmap(updated_est_depth_bev[i].detach().cpu().numpy(), cbar=True, ax=axes[1], vmin=0, vmax=1)
                sns.heatmap(gt_depth_valid_mask[i].detach().cpu().numpy(), cbar=True, ax=axes[2], vmin=0, vmax=1)
                sns.heatmap(gt_depth_fg_mask[i].detach().cpu().numpy(), cbar=True, ax=axes[3], vmin=0, vmax=1)
                sns.heatmap(est_obj_bev[i].detach().cpu().numpy(), cbar=True, ax=axes[4], vmin=0, vmax=1)

                for ax_i in range(5):
                    axes[ax_i].set_axis_off()
                fig.tight_layout()
                
                plt.savefig(os.path.join(save_dir, 'bevcoord_{}.png'.format(i)))
                plt.close()
            self.vis_count += 1
                
        return output_dict

    def _forward_ss(self, data_dict):
        # x:[sum(record_len), 4, 3or4, 256, 352]
        # rots: [sum(record_len), 4, 3, 3]
        # trans: [sum(record_len), 4, 3]
        # intrins: [sum(record_len), 4, 3, 3]
        # post_rots: [sum(record_len), 4, 3, 3]
        # post_trans: [sum(record_len), 4, 3]
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: sum(record_len) x C x 240 x 240 (4 x 64 x 240 x 240)
        
        x = self.bevencode(x)  # 用resnet18提取特征  x: sum(record_len) x C x 240 x 240
        if self.shrink_flag:
            x = self.shrink_conv(x)
        # 4 x C x 120 x 120

        ## fusion ##
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x_fuse = self.fusion(x, record_len, pairwise_t_matrix)
        ############

        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'depth_items': depth_items}

        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dm": dm})

        if self.supervise_single:
            psm_single = self.cls_head(x)
            rm_single = self.reg_head(x)
            output_dict.update({'psm_single': psm_single,
                                'rm_single': rm_single})
            if self.use_dir:
                dm_single = self.dir_head(x)
                output_dict.update({"dm_single": dm_single})

        return output_dict


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShootCollaSmoothfg(grid_conf, data_aug_conf, outC)