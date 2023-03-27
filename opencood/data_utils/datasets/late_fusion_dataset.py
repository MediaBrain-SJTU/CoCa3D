# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for late fusion
"""
import random
import math
from collections import OrderedDict

import numpy as np
import os
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from opencood.utils import pcd_utils

import opencood.data_utils.datasets
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import read_json
# from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
from opencood.utils.pose_utils import add_noise_data_dict

class LateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(LateFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

        if "box_align" in params.keys():
            self.box_align = True
            self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
            self.stage1_result = read_json(self.stage1_result_path)
            self.box_align_args = params['box_align']['args']
        
        else:
            self.box_align = False

        # 2022.10.14, remove it later
        # OPV2V More Agents, reinitialize the scenario base
        if "valid_data" in params:
            assert "MoreAgents" in self.root_dir
            data_info = self.load_pkl_files(params["valid_data"])
            if train:
                data_info = data_info['train']

            else:
                if "test" in params['validate_dir']:
                    data_info = data_info['test']
                else:
                    data_info = data_info['validate']
            self._init_database(data_info, self.root_dir)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict, idx)

        return reformat_data_dict

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

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
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def get_item_single_car(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)

        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                    selected_cav_base[
                                                           'params'][
                                                           'lidar_pose_clean'])

        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})

        return selected_cav_processed

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()
        base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])
        # during training, we return a random cav's data
        # only one vehicle is in processed_data_dict
        if not self.visualize:
            selected_cav_id, selected_cav_base = \
                random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict, idx):
        """
            processed_data_dict.keys() = ['ego', "650", "659", ...]
        """
        base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []
        cav_id_list = []
        lidar_pose_list = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > self.params['comm_range']:
                continue
            cav_id_list.append(cav_id)
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])

        ########## Added by Yifan Lu 2022.8.14 ##############
        # box align to correct pose.
        if self.box_align and str(idx) in self.stage1_result.keys():
            stage1_content = self.stage1_result[str(idx)]
            if stage1_content is not None:
                cav_id_list_stage1 = stage1_content['cav_id_list']
                
                pred_corners_list = stage1_content['pred_corner3d_np_list']
                pred_corners_list = [np.array(corners, dtype=np.float64) for corners in pred_corners_list]
                uncertainty_list = stage1_content['uncertainty_np_list']
                uncertainty_list = [np.array(uncertainty, dtype=np.float64) for uncertainty in uncertainty_list]
                stage1_lidar_pose_list = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list_stage1]
                stage1_lidar_pose = np.array(stage1_lidar_pose_list)


                refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                stage1_lidar_pose, 
                                                                uncertainty_list=uncertainty_list, 
                                                                **self.box_align_args)
                stage1_lidar_pose[:,[0,1,4]] = refined_pose
                stage1_lidar_pose_refined_list = stage1_lidar_pose.tolist() # updated lidar_pose_list
                for cav_id, lidar_pose_refined in zip(cav_id_list_stage1, stage1_lidar_pose_refined_list):
                    if cav_id not in cav_id_list:
                        continue
                    idx_in_list = cav_id_list.index(cav_id)
                    lidar_pose_list[idx_in_list] = lidar_pose_refined
                    base_data_dict[cav_id]['params']['lidar_pose'] = lidar_pose_refined

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]
            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
            cav_lidar_pose_clean = selected_cav_base['params']['lidar_pose_clean']
            transformation_matrix_clean = x1_to_x2(cav_lidar_pose_clean, ego_lidar_pose_clean)

            selected_cav_processed = \
                self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix': transformation_matrix,
                                           'transformation_matrix_clean': transformation_matrix_clean})
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]

                if (self.params['only_vis_ego'] is False) or (cav_id=='ego'):
                    print(cav_id)
                    import copy
                    projected_lidar = copy.deepcopy(cav_content['origin_lidar'])
                    projected_lidar[:, :3] = \
                        box_utils.project_points_by_matrix_torch(
                            projected_lidar[:, :3],
                            transformation_matrix)
                    projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix'])).float()
            
            # late fusion training, no noise
            transformation_matrix_clean_torch = transformation_matrix_torch

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'transformation_matrix_clean': transformation_matrix_clean_torch})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        if self.visualize:
            projected_lidar_stack = [torch.from_numpy(
                np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})
            # output_dict['ego'].update({'projected_lidar_list': projected_lidar_list})

        return output_dict

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
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor


    def post_process_no_fusion(self, data_dict, output_dict_ego, return_uncertainty=False):
        data_dict_ego = OrderedDict()
        data_dict_ego['ego'] = data_dict['ego']
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        if return_uncertainty:
            pred_box_tensor, pred_score, uncertainty = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego, return_uncertainty=True)
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty
        else:
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict_ego, output_dict_ego)
            return pred_box_tensor, pred_score, gt_box_tensor

