# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for early fusion
"""
import random
import math
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.datasets import intermediate_fusion_dataset_v2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, tfm_to_pose

from icecream import ic


class IntermediateFusionDatasetV2V2X(intermediate_fusion_dataset_v2.IntermediateFusionDatasetV2):
    """
        IntermediateFusionDatasetV2V2X is children of IntermediateFusionDatasetV2
        It works almost the same as IntermediateFusionDataset,
        But it 
        rewrite __init__, __len__, retrieve_base_data , which is from BaseDataset
        rewrite generate_object_center, which is from IntermediateFusionDataset
    """

    ### rewrite __init__ ###
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None
            
        # same as IntermediateFusionDataset
        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
            
        if "kd_flag" in params.keys():
            self.kd_flag = params['kd_flag']
        else:
            self.kd_flag = False

        if "box_align" in params.keys():
            self.box_align = True
        else:
            self.box_align = False
        
        print("Dataset dir:", root_dir)

        # v2x: read the pickle file
        with open(root_dir, 'rb') as f:
            dataset_infos = pickle.load(f)  # dataset_infos is a list 

        self.agent_start = eval(min([i[-1] for i in dataset_infos[0].keys() if i.startswith("lidar_pose")]))
        # self.agent_start = 1 when v2x-sim2.0,  = 0 when v2x-sim1.0


        if 'train_params' not in params or\
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.keyframe_database = OrderedDict()
        self.len_record = len(dataset_infos)

        # loop over all keyframe.
        # data_info is one sample.
        # np.random.seed(303)
        for (i, data_info) in enumerate(dataset_infos):
            self.keyframe_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_num = data_info['agent_num']
            assert cav_num > 0
            select_ego = 1

            # in one keyframe, loop all agent
            for cav_id in range(self.agent_start, self.agent_start + cav_num):

                self.keyframe_database[i][cav_id] = OrderedDict()
                self.keyframe_database[i][cav_id]['lidar'] = data_info[f'lidar_path_{cav_id}']  # maybe add camera in the future
                self.keyframe_database[i][cav_id]['params'] = OrderedDict()
                self.keyframe_database[i][cav_id]['params']['lidar_pose'] = tfm_to_pose(data_info[f"lidar_pose_{cav_id}"]) # tfm in data_info, turn to [x,y,z,roll,yaw,pitch]
                self.keyframe_database[i][cav_id]['params']['vehicles'] = data_info[f'labels_{cav_id}']['gt_boxes_global']
                self.keyframe_database[i][cav_id]['params']['object_ids'] = data_info[f'labels_{cav_id}']['gt_object_ids'].tolist()

                # randomly select ego is better
                if cav_id == select_ego:
                    self.keyframe_database[i][cav_id]['ego'] = True
                    
                else:
                    self.keyframe_database[i][cav_id]['ego'] = False

    ### rewrite __len__ ###
    def __len__(self):
        return self.len_record

    ### rewrite retrieve_base_data ###
    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
            lidar_np: (N, 4)
        """
        # we loop the accumulated length list to see get the scenario index
        keyframe = self.keyframe_database[idx]

        data = OrderedDict()
        ego_idx = -1
        # load files for all CAVs
        for cav_id, cav_content in keyframe.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            if cav_content['ego']:
                ego_idx = cav_id
            data[cav_id]['params'] = cav_content['params'] # lidar_pose, vehicles(gt_boxes), object_id(token)

            # load the corresponding data into the dictionary
            nbr_dims = 4 # x,y,z,intensity
            scan = np.fromfile(cav_content['lidar'], dtype='float32')
            points = scan.reshape((-1, 5))[:, :nbr_dims] 
            data[cav_id]['lidar_np'] = points

        # in fact we want to put ego in the first place
        # we will exchange two agent.
        if ego_idx != 0:
            if self.agent_start == 0:  # v2x-sim1.0
                data[ego_idx], data[0] = data[0], data[ego_idx]
            elif self.agent_start == 1:  # v2x-sim2.0
                data[0] = data[ego_idx]
                data.move_to_end(0, last=False) # move to first
                data.pop(ego_idx)

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

        return self.post_processor.generate_object_center_v2x(cav_contents,
                                                        reference_lidar_pose)