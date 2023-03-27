# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Basedataset class for all kinds of fusion.
"""

import yaml
from concurrent.futures import process
import os
from collections import OrderedDict
import cv2
import pickle as pkl
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import sys
from PIL import Image
from tqdm import tqdm
import multiprocessing
import json
import time
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.utils.keypoint_utils import bev_sample, get_keypoints
from opencood.utils.camera_utils import load_camera_data

class CameraBaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp
    and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """

    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        if 'select_kp' in params:
            self.select_keypoint = params['select_kp']
        else:
            self.select_keypoint = None

        assert 'proj_first' in params['fusion']['args']
        if params['fusion']['args']['proj_first']:
            self.proj_first = True
        else:
            self.proj_first = False
        
        self.fg_mask = False
        if 'fg_mask' in params:
            self.fg_mask = params['fg_mask']

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir 
        
        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        # intermediate and supervise single
        self.supervise_single = True \
            if ('supervise_single' in params['train_params'] and params['train_params']['supervise_single']) \
            else False

        # load images to memory
        self.preload = True \
            if ('preload' in params['train_params'] and params['train_params']['preload']) \
            else False

        if self.preload:
            self.preload_worker_num = params['train_params']['preload']['mp']
            if self.preload_worker_num:
                self.preload_shared_dict = multiprocessing.Manager().dict()
            self.all_base_data = []
            

        # depth gt
        self.use_gt_depth = True \
            if ('camera_params' in params and params['camera_params']['use_depth_gt']) \
            else False
        self.use_fg_mask = True \
            if ('use_fg_mask' in params['loss']['args'] and params['loss']['args']['use_fg_mask']) \
            else False


        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        scenario_folders_name = sorted([x
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml')])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)
                    depth_files = self.load_depth_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
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

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

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
        loader = yaml.Loader
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

        # random input
        cav_id_list = list(scenario_database.keys())
        cav_id_list_ = cav_id_list[1:]
        random.shuffle(cav_id_list_)
        cav_id_list = cav_id_list[:1] + cav_id_list_

        for cav_id in cav_id_list:
            cav_content = scenario_database[cav_id]
            ####### OPV2V MoreAgents ######
            if timestamp_key not in cav_content:
                continue
            ###############################
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            # JSON is faster reading
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['params'] = json.load(f)
            else:
                data[cav_id]['params'] = \
                    load_yaml(cav_content[timestamp_key]['yaml'])

            ### OPV2V_moreagent do not have lidar ###
            # only load when needing visualization.
            if self.visualize:
                if not os.path.exists(cav_content[timestamp_key]['lidar']):
                    data[cav_id]['lidar_np'] = np.zeros((1,4))
                else:
                    data[cav_id]['lidar_np'] = \
                        pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

            if self.select_keypoint and 'lidar_keypoints_np' in cav_content[timestamp_key]:
                data[cav_id]['lidar_keypoints_np'] = cav_content[timestamp_key]['lidar_keypoints_np']

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

    def retrieve_all_base_data(self):
        time1 = time.time()
        for idx in tqdm(range(self.__len__())):
            self.all_base_data.append(self.retrieve_base_data(idx))
        time2 = time.time()
        print("retrieve time usage:", time2-time1)

    def retrieve_per_process(self, proc_id, idx_list, lock):
        base_data_list = []
        for idx in tqdm(idx_list):
            base_data_list.append(self.retrieve_base_data(idx))
        # with lock:
        self.preload_shared_dict[proc_id] = base_data_list
        # print(f"Proc {proc_id} has object of memory {sys.getsizeof(base_data_list)/1024/1024} MB.")

    def retrieve_all_base_data_mp(self):
        data_num = self.__len__()
        print("len:", data_num)

        lock = multiprocessing.Lock()
        idx_list_split = np.array_split(list(range(data_num)), self.preload_worker_num)
        idx_list_split = [x.tolist() for x in idx_list_split]
        process_pool = []
        for i in range(self.preload_worker_num):
            p = multiprocessing.Process(target=self.retrieve_per_process, args=(i, idx_list_split[i], lock))
            process_pool.append(p)
        time1 = time.time()
        for p in process_pool:
            p.start()
        for p in process_pool:
            p.join()
        time2 = time.time()
        print("retrieve time usage:", time2-time1)
        for i in range(self.preload_worker_num):
            self.all_base_data.extend(self.preload_shared_dict[i])
        self.preload_shared_dict.clear() # release memory
        time3 = time.time()
        print("merge time usage:", time3-time2)
        # with open("/GPFS/rhome/yifanlu/OpenCOOD/vis_result/retrieve_100_base_data.pkl", "wb") as f:
        #     pickle.dump(self.all_base_data, f)

        print("len:",len(self.all_base_data))

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    @staticmethod
    def load_depth_files(cav_path, timestamp):
        """
        Retrieve the paths to all depth files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        depth0_file = os.path.join(cav_path,
                                    timestamp + '_depth0.png')
        depth1_file = os.path.join(cav_path,
                                    timestamp + '_depth1.png')
        depth2_file = os.path.join(cav_path,
                                    timestamp + '_depth2.png')
        depth3_file = os.path.join(cav_path,
                                    timestamp + '_depth3.png')
        return [depth0_file, depth1_file, depth2_file, depth3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for early and late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'anchor_box': torch.from_numpy(ego_dict['anchor_box']),
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)

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
            assert len(cav_list) > 0

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs _init_database')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                timestamps = data_info[scenario_name][cav_id]

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    yaml_file = yaml_file.replace('OPV2V_MoreAgents', \
                                                  'OPV2V_MoreAgents/dataset_annos')

                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')

                    camera_files = self.load_camera_files(cav_path, timestamp)
                    depth_files = self.load_depth_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
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