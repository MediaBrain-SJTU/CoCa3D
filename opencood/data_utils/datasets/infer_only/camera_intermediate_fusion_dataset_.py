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
from icecream import ic
from PIL import Image
import h5py
import os
import json
import pickle as pkl
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import camera_intermediate_fusion_dataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
    gen_dx_bx,
    load_camera_data
)
from opencood.utils.pose_utils import add_noise_data_dict

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data
    
class CameraIntermediateFusionDataset_(camera_intermediate_fusion_dataset.CameraIntermediateFusionDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(CameraIntermediateFusionDataset_, self).__init__(params, visualize, train)

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
            if(len(cav_list) == 0):
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
                    #############################################
                    # for lidar visible GT
                    # if j == 0:
                    #     yaml_file = yaml_file.replace('OPV2V_MoreAgents', 'OPV2V')
                    ##############################################
                    
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
            if(len(data) == self.max_cav):
                break
        return data

    # also need to rewrite __getitem__
    # when infer, we just need ego's all gt
    # not merged gt.
    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []
        ego_cav_base = None

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                ego_cav_base = cav_content
                break
            
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0


        agents_image_inputs = []
        object_stack = []
        object_id_stack = []
        single_label_list = []
        too_far = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        projected_lidar_clean_list = []
        cav_id_list = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)

            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue

            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)   

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]

            selected_cav_processed = self.get_item_single_car_camera(
                selected_cav_base,
                ego_cav_base)
            
            ############# Modified in CameraIntermediateFusionDataset_ ###########
            if ego_id == cav_id:
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']

            agents_image_inputs.append(
                selected_cav_processed['image_inputs'])

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
            
            if self.supervise_single:
                single_label_list.append(selected_cav_processed['single_label_dict'])

        ########## Added by Yifan Lu 2022.10.10 ##############
        # generate single view GT label
        if self.supervise_single:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                processed_data_dict['ego'].update(
                    {"single_label_dict_torch": single_label_dicts}
                )

        ########## Added by Yifan Lu 2022.4.5 ################
        # filter those out of communicate range
        # then we can calculate get_pairwise_transformation
        for cav_id in too_far:
            base_data_dict.pop(cav_id)
        
        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        ######################################################
        
        # exclude all repetitive objects    
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(agents_image_inputs)

        merged_image_inputs_dict = self.merge_features_to_dict(agents_image_inputs, merge='stack')

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=self.anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'image_inputs': merged_image_inputs_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'lidar_poses_clean': lidar_poses_clean,
             'lidar_poses': lidar_poses})


        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})


        processed_data_dict['ego'].update({'sample_idx': idx,
                                            'cav_id_list': cav_id_list})

        return processed_data_dict