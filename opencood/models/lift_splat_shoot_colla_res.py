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

class LiftSplatShootCollaRes(LiftSplatShoot):
    def __init__(self, args): 
        super(LiftSplatShootCollaRes, self).__init__(args)

        fusion_args = args['fusion_args']
        self.ms = args['fusion_args']['core_method'].endswith("ms")
        if self.ms:
            self.bevencode = BevEncodeMSFusion(fusion_args)
        else:
            self.fusion = FusionNet(fusion_args)
        self.supervise_single = args['supervise_single']

        self.colla_depth = CollaDepthNet(dim=128, downsample_rate=1, discrete_ratio=fusion_args['args']['voxel_size'][0])
    
    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        depth_logit, depth_gt_indices, depth_gt, ori_x, x = self.camencode(x) # 进行图像编码 ori_x: B*N x C x fH x fW, x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        
        ori_x = ori_x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        ori_x = ori_x.unsqueeze(3).expand(-1, -1, -1, self.D, -1, -1)
        ori_x = ori_x.permute(0, 1, 3, 4, 5, 2)

        depth_prob = F.softmax(depth_logit, dim=1)
        depth_prob = depth_prob.view(B, N, 1, self.D, imH//self.downsample, imW//self.downsample)
        depth_prob = depth_prob.permute(0, 1, 3, 4, 5, 2)

        depth_gt = depth_gt.view(B, N, 1, self.D, imH//self.downsample, imW//self.downsample)
        depth_gt = depth_gt.permute(0, 1, 3, 4, 5, 2)

        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)
        return ori_x, x, (depth_logit,depth_gt_indices), (depth_prob, depth_gt)

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
        x, x_new, depth_items, depth_items_global = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
        x_new = self.voxel_pooling(geom, x_new)
        depth_probs = self.voxel_pooling(geom, depth_items_global[0])
        depth_gt = self.voxel_pooling(geom, depth_items_global[1].float())
        
        # Multi-view matching & Tune depth probs
        updated_depth_probs = self.colla_depth(x, record_len, data_dict['pairwise_t_matrix'], depth_probs)
        x = x * updated_depth_probs + x_new

        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x_single, x_fuse = self.bevencode(x, record_len, pairwise_t_matrix)
        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'depth_items': depth_items}
        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dm": dm})

        if self.supervise_single:
            psm_single = self.cls_head(x_single)
            rm_single = self.reg_head(x_single)
            output_dict.update({'psm_single': psm_single,
                                'rm_single': rm_single})
            if self.use_dir:
                dm_single = self.dir_head(x_single)
                output_dict.update({"dm_single": dm_single})

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
    return LiftSplatShootCollaRes(grid_conf, data_aug_conf, outC)