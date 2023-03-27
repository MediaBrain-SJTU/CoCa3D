# testing multiview camera dataset

"""
pure camera api, remove codes of LiDAR
"""
from locale import str
from builtins import enumerate, len, list
import random
import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
import copy
from icecream import ic
import os
from PIL import Image
import h5py
import pickle as pkl
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import camera_late_fusion_dataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.camera_utils import (
    load_camera_data,
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
    gen_dx_bx,
)
import json
def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


class CameraLateFusionDataset_(camera_late_fusion_dataset.CameraLateFusionDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(CameraLateFusionDataset_, self).__init__(params, visualize, train)

    # when infer, use ego's all gt. do not need visibility map
    def generate_object_center(
        self, cav_contents, reference_lidar_pose, visibility_map
    ):
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
        
        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(
            cav_contents, reference_lidar_pose, enlarge_z=True
        )

    @staticmethod
    def load_pkl_files(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        return data

    def _init_database(self, data_info, root_dir):
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in data_info if
                                   os.path.isdir(os.path.join(root_dir, x))])
    
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            scenario_name = scenario_folder.split('/')[-1]
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted(data_info[scenario_name])
            if len(cav_list) == 0:
                continue
            assert len(cav_list) > 0


            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                timestamps = data_info[scenario_name][cav_id]

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    yaml_file2 = yaml_file.replace("OPV2V_MoreAgents", "OPV2V_MoreAgents/dataset_annos")

                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    lidar_file = lidar_file.replace("OPV2V_MoreAgents", "OPV2V")

                    camera_files = self.load_camera_files(cav_path, timestamp)
                    depth_files = self.load_depth_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['yaml2'] = \
                        yaml_file2
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = \
                        depth_files

                   # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name                  

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

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
        """
        from opencood.utils import pcd_utils
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            ####### OPV2V MoreAgents ######
            if timestamp_key not in cav_content:
                continue
            ###############################
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            data[cav_id]['params'] = \
                load_json(cav_content[timestamp_key]['yaml'].replace("yaml", "json"))
            data[cav_id]['params']['lidar_pose'] = \
                load_json(cav_content[timestamp_key]['yaml2'].replace("yaml", "json"))['lidar_pose']
                
            ### OPV2V_moreagent do not have lidar ###
            if self.visualize:
                if not os.path.exists(cav_content[timestamp_key]['lidar']):
                    data[cav_id]['lidar_np'] = np.zeros((1,4))
                else:
                    data[cav_id]['lidar_np'] = \
                        pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

            hdf5_file = cav_content[timestamp_key]['cameras'][0].replace("camera0.png", "imgs_hdf5")
            if os.path.exists(hdf5_file):
                # faster reading
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]['camera_data'] = []
                    data[cav_id]['depth_data'] = []
                    for i in range(4):
                        data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                        data[cav_id]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))

            else:
                data[cav_id]['camera_data'] = \
                    load_camera_data(cav_content[timestamp_key]['cameras'], self.preload)
                if self.use_gt_depth:  
                    data[cav_id]['depth_data'] = \
                        load_camera_data(cav_content[timestamp_key]['depths'], self.preload) # we can use the same loading api

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("train","additional/train")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("validate","additional/validate")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("test","additional/test")
                    
                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = \
                        cv2.imread(cav_content[timestamp_key][file_extension])
            if len(data) == self.max_cav:
                break
        return data

    # only use ego's all gt boxes.
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
        gt_box_tensor = self.post_processor.generate_gt_bbx({"ego": data_dict['ego']})

        return pred_box_tensor, pred_score, gt_box_tensor