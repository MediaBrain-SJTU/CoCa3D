# -*- coding: utf-8 -*-
# Author: Qunahao Li

"""
Dataset class for DAIR-V2X dataset late fusion
"""
import random
import math
from collections import OrderedDict
import os
import opencood.data_utils.post_processor as post_processor
import numpy as np
import torch
from torch.utils.data import DataLoader
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
import opencood.data_utils.datasets
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import late_fusion_dataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2
import json
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data
class LateFusionDatasetDAIR(late_fusion_dataset.LateFusionDataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        self.max_cav = 2
        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False


        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
        
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None
        
        if "box_align" in params.keys():
            self.box_align = True
            self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
            self.stage1_result = load_json(self.stage1_result_path)
            self.box_align_args = params['box_align']['args']
        
        else:
            self.box_align = False

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']

        self.split_info = load_json(split_dir)
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

    def __len__(self):
        return len(self.split_info)

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False
                
        data[0]['params'] = OrderedDict()
        data[0]['params']['vehicles'] = load_json(os.path.join(self.root_dir, 'vehicle-side/label/lidar/',veh_frame_id + '.json'))
        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))

        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,novatel_to_world_json_file)

        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))

        data[1]['params'] = OrderedDict()
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")

        data[1]['params']['vehicles'] = load_json(os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/',inf_frame_id + '.json'))
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix1 = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file,system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix1)

        data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))
        return data

    ### rewrite generate_object_center ###
    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_dairv2x_single(cav_contents)

    ### rewrite post_process ###
    """
    We have to rewrite post_process for LateFusionDatasetDAIR
    because the object id can not used for identifying the same object
    
    here we will to use the IoU to determine it.
    """
    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    ### rewrite post_process_no_fusion ###
    """
    We have to rewrite post_process for LateFusionDatasetDAIR
    because the object id can not used for identifying the same object
    
    here we will to use the IoU to determine it.
    """
    def post_process_no_fusion(self, data_dict, output_dict_ego, return_uncertainty=False):
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx_by_iou(data_dict)

        if return_uncertainty:
            pred_box_tensor, pred_score, uncertainty = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty
        else:
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego)
            return pred_box_tensor, pred_score, gt_box_tensor