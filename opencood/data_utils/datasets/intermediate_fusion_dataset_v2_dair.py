# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for early fusion
"""
import random
import math
from collections import OrderedDict

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils, common_utils

from opencood.data_utils.datasets import intermediate_fusion_dataset_v2
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import x_to_world
from opencood.utils.pose_utils import add_noise_data_dict

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data

class IntermediateFusionDatasetV2DAIR(intermediate_fusion_dataset_v2.IntermediateFusionDatasetV2):
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

        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False

        if "box_align" in params.keys():
            self.box_align = True
            self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
            self.stage1_result = load_json(self.stage1_result_path)
            self.box_align_args = params['box_align']['args']
        
        else:
            self.box_align = False

        assert 'clip_pc' in params['fusion']['args']
        if params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False
        
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None

        self.order= params['postprocess']['order']
        
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

    def __len__(self):
        return len(self.split_info)

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



    ### rewrite __getitem___
    ### we should use iou to filter repetitive boxes, instead of object id.
    def __getitem__(self, idx):
        # put here to avoid initialization error
        from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils \
            import points_in_boxes_cpu
        base_data_dict = self.retrieve_base_data(idx)
        base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        processed_features = []
        object_stack = []
        object_id_stack = []
        cav_object_stack = []
        lidar_pose_stack = []
        lidar_pose_clean_stack = []


        projected_lidar_stack = []
        no_projected_lidar_stack = []


        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > self.params['comm_range']:
                continue

            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose,
                ego_lidar_pose_clean)
            if len(selected_cav_processed['projected_lidar']) > 10:
                object_stack.append(
                    selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                
                cav_object_stack.append(selected_cav_processed['cav_object_bbx_center'])

                processed_features.append(
                    selected_cav_processed['processed_features'])


                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

                no_projected_lidar_stack.append(
                    selected_cav_processed['no_projected_lidar'])

                lidar_pose_stack.append(
                        selected_cav_base['params']['lidar_pose'])

                lidar_pose_clean_stack.append(
                    selected_cav_base['params']['lidar_pose_clean'])
                
        if self.visualize:
            projected_lidar_stack = projected_lidar_stack

        # exclude all repetitive objects for cooperative label
        # DAIR-V2X only has two agent. And cav_object_stack
        # Modified by Yifan Lu 2022.09.24
        if len(object_stack) == 1:
            object_stack_all = object_stack[0]
        else:
            veh_corners_np = box_utils.boxes_to_corners_3d(object_stack[0], self.order)
            inf_corners_np = box_utils.boxes_to_corners_3d(object_stack[1], self.order)
            # inf corners are in inf-coord, should transform to veh-coord
            inf_to_veh = x1_to_x2(lidar_pose_clean_stack[1], lidar_pose_clean_stack[0])
            inf_corners_np = box_utils.project_box3d(inf_corners_np, inf_to_veh)

            inf_polygon_list = list(common_utils.convert_format(inf_corners_np))
            veh_polygon_list = list(common_utils.convert_format(veh_corners_np))
            iou_thresh = 0.05 

            gt_from_inf = []
            for i in range(len(inf_polygon_list)):
                inf_polygon = inf_polygon_list[i]
                ious = common_utils.compute_iou(inf_polygon, veh_polygon_list)
                if (ious > iou_thresh).any():
                    continue
                gt_from_inf.append(inf_corners_np[i])
            
            if len(gt_from_inf):
                gt_from_inf = np.stack(gt_from_inf)
                co_corners_np = np.vstack([veh_corners_np, gt_from_inf])
            else:
                co_corners_np = veh_corners_np
            object_stack_all = box_utils.corner_to_center(co_corners_np, order=self.order)


        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack_all.shape[0], :] = object_stack_all
        mask[:object_stack_all.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center, # hwl
                anchors=anchor_box,
                mask=mask)

        # Filter empty boxes
        object_stack_filtered = []
        label_dict_no_coop = []
        cav_label_dict_no_coop = []
        idx_cnt = 0
        for boxes, points in zip(object_stack, projected_lidar_stack):
            point_indices = points_in_boxes_cpu(points[:, :3], boxes[:,
                                                               [0, 1, 2, 5, 4,
                                                                3, 6]])
            cur_mask = point_indices.sum(axis=1) > 0

            # added by yifanlu
            cav_itself_object_center = np.array(cav_object_stack[idx_cnt])
            bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            bbx_mask = np.zeros(self.params['postprocess']['max_num'])
            bbx_center[:cav_itself_object_center.shape[0], :] = cav_itself_object_center
            bbx_mask[:cav_itself_object_center.shape[0]] = 1

            cav_label_dict_no_coop.append(
                self.post_processor.generate_label(
                    gt_box_center=bbx_center, # hwl
                    anchors=anchor_box,
                    mask=bbx_mask)
            )
            idx_cnt += 1
            ###

            if cur_mask.sum() == 0:
                label_dict_no_coop.append({
                    'pos_equal_one': np.zeros((*anchor_box.shape[:2],
                                               self.post_processor.anchor_num)),
                    'neg_equal_one': np.ones((*anchor_box.shape[:2],
                                              self.post_processor.anchor_num)),
                    'targets': np.zeros((*anchor_box.shape[:2],
                                         self.post_processor.anchor_num * 7))
                })
                continue
            object_stack_filtered.append(boxes[cur_mask])
            bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            bbx_mask = np.zeros(self.params['postprocess']['max_num'])
            bbx_center[:boxes[cur_mask].shape[0], :] = boxes[cur_mask]
            bbx_mask[:boxes[cur_mask].shape[0]] = 1
            label_dict_no_coop.append(
                self.post_processor.generate_label(
                    gt_box_center=bbx_center, # hwl
                    anchors=anchor_box,
                    mask=bbx_mask)
            )

        if not self.proj_first:
            label_dict_no_coop = cav_label_dict_no_coop

        label_dict = {
            'stage1': label_dict_no_coop,
            'stage2': label_dict
        }
        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': list(range(object_stack_all.shape[0])), # meaningless
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'lidar_poses_clean': np.stack(lidar_pose_clean_stack),
             'lidar_poses': np.stack(lidar_pose_stack),
             'pairwise_t_matrix':self.get_pairwise_transformation(lidar_pose_stack, self.max_cav)})

        if self.proj_first:
            processed_data_dict['ego'].update({'origin_lidar':
                                                projected_lidar_stack})
        else:
            processed_data_dict['ego'].update({'origin_lidar':
                                                no_projected_lidar_stack})
        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar_vis':
                                                np.vstack(
                                                    projected_lidar_stack)})
        return processed_data_dict
