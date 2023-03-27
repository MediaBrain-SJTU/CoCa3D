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
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum
from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, Up, CamEncode, CamEncodeGTDepth, BevEncode
from opencood.models.sub_modules.downsample_conv import DownsampleConv
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

class LiftSplatShootIntermediate(LiftSplatShoot):
    def __init__(self, args): 
        super(LiftSplatShootIntermediate, self).__init__(args)

        fusion_args = args['fusion_args']
        self.ms = args['fusion_args']['core_method'].endswith("ms")
        if self.ms:
            self.bevencode = BevEncodeMSFusion(fusion_args)
        else:
            self.fusion = FusionNet(fusion_args)
        self.supervise_single = args['supervise_single']

    
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
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: sum(record_len) x C x 240 x 240 (4 x 64 x 240 x 240)
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
    return LiftSplatShootIntermediate(grid_conf, data_aug_conf, outC)