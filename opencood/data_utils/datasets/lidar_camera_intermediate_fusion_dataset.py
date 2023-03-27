# Author: Yifan Lu
"""
hybrid lidar and camera dataset
"""
import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
from icecream import ic
import os
import pickle as pkl
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import camera_basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils import box_utils
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
    gen_dx_bx,
)
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.utils.common_utils import read_json
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)


class LiDARCameraIntermediateFusionDataset(camera_basedataset.CameraBaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(LiDARCameraIntermediateFusionDataset, self).__init__(params, visualize, train)
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]
        self.grid_conf = params["fusion"]["args"]["grid_conf"]
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)

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
                load_yaml(cav_content[timestamp_key]['yaml'])

            ### OPV2V_moreagent do not have lidar ###
            # go to OPV2V for lidar
            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

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

        return data

    def get_item_single_car(self, selected_cav_base, ego_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.


        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
            including 'params', 'camera_data'
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose) # T_ego_cav
        transformation_matrix_clean = \
            x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                     ego_pose_clean)
        
        # process lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        # x,y,z in ego space
        projected_lidar = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                        transformation_matrix)
        if self.proj_first:
            lidar_np[:, :3] = projected_lidar

        if self.visualize:
            # filter lidar
            selected_cav_processed.update({'projected_lidar': projected_lidar})
            
        processed_lidar = self.pre_processor.preprocess(lidar_np)


        # generate targets label single GT
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], selected_cav_base['params']['lidar_pose']
        )
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=anchor_box, mask=object_bbx_mask
        )
        selected_cav_processed.update({"single_label_dict": label_dict})


        ######## Camera #########
        # adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py
        camera_data_list = selected_cav_base["camera_data"]

        params = selected_cav_base["params"]
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for idx, img in enumerate(camera_data_list):
            camera_coords = np.array(params["camera%d" % idx]["cords"]).astype(
                np.float32
            )
            camera_to_lidar = x1_to_x2(
                camera_coords, params["lidar_pose_clean"]
            ).astype(np.float32)  # T_LiDAR_camera
            camera_to_lidar = camera_to_lidar @ np.array(
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32)  # UE4 coord to opencv coord
            # lidar_to_camera = np.array(params['camera%d' % idx]['extrinsic']).astype(np.float32) # Twc^-1 @ Twl = T_camera_LiDAR
            camera_intrinsic = np.array(params["camera%d" % idx]["intrinsic"]).astype(
                np.float32
            )

            intrin = torch.from_numpy(camera_intrinsic)
            rot = torch.from_numpy(
                camera_to_lidar[:3, :3]
            )  # R_wc, we consider world-coord is the lidar-coord
            tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            img_src = [img]

            # depth
            if self.use_gt_depth:
                depth_img = selected_cav_base["depth_data"][idx]
                img_src.append(depth_img)
            else:
                depth_img = None

            if self.use_fg_mask:
                _, _, fg_mask = coord_3d_to_2d(
                                box_utils.boxes_to_corners_3d(object_bbx_center[:len(object_ids)], self.params['postprocess']['order']),
                                camera_intrinsic,
                                camera_intrinsic) 
                fg_mask = np.array(fg_mask*255, dtype=np.uint8)
                fg_mask = Image.fromarray(fg_mask)
                img_src.append(fg_mask)


            # data augmentation
            resize, resize_dims, crop, flip, rotate = sample_augmentation(
                self.data_aug_conf, self.train
            )
            img_src, post_rot2, post_tran2 = img_transform(
                img_src,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # decouple RGB and Depth

            img_src[0] = normalize_img(img_src[0])
            if self.use_gt_depth:
                img_src[1] = img_to_tensor(img_src[1]) * 255
            if self.use_fg_mask
                img_src[-1] = img_to_tensor(img_src[-1])

            imgs.append(torch.cat(img_src, dim=0))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        selected_cav_processed.update(
            {
            "image_inputs": 
                {
                    "imgs": torch.stack(imgs), # [Ncam, 3or4, H, W]
                    "intrins": torch.stack(intrins),
                    "rots": torch.stack(rots),
                    "trans": torch.stack(trans),
                    "post_rots": torch.stack(post_rots),
                    "post_trans": torch.stack(post_trans),
                }
            }
        )

        # anchor box
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({"anchor_box": anchor_box})

        ######## LiDAR ##########
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                    ego_pose_clean)

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
                'projected_lidar': projected_lidar,
                'processed_features': processed_lidar,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean
            }
        )




        return selected_cav_processed

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
        processed_features = []
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
                
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                selected_cav_processed['processed_features'])
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
        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        
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
        merged_feature_dict = self.merge_features_to_dict(processed_features)
        merged_image_inputs_dict = self.merge_features_to_dict(agents_image_inputs, merge='stack')

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'image_inputs': merged_image_inputs_dict,
             'processed_lidar': merged_feature_dict,
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

    @staticmethod
    def merge_features_to_dict(processed_feature_list, merge=None):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.
        merge : "stack" or "cat". used for images

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
        
        # stack them
        # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
        # cat them
        # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
        if merge=='stack': 
            for feature_name, features in merged_feature_dict.items():
                merged_feature_dict[feature_name] = torch.stack(features, dim=0)
        elif merge=='cat':
            for feature_name, features in merged_feature_dict.items():
                merged_feature_dict[feature_name] = torch.cat(features, dim=0)

        return merged_feature_dict


    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        image_inputs_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        
        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        if self.visualize:
            origin_lidar = []
        
        ### 2022.10.10 single gt ####
        if self.supervise_single:
            pos_equal_one_single = []
            neg_equal_one_single = []
            targets_single = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

            processed_lidar_list.append(ego_dict['processed_lidar'])
            image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
            record_len.append(ego_dict['cav_num'])

            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                targets_single.append(ego_dict['single_label_dict_torch']['targets'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        # {"image_inputs": 
        #   {image: [sum(record_len), Ncam, C, H, W]}
        # }
        merged_image_inputs_dict = self.merge_features_to_dict(image_inputs_list, merge='cat')
        
        # [sum(record_len), C, H, W]
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)

        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'image_inputs': merged_image_inputs_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'lidar_pose_clean': lidar_pose_clean,
                                   'lidar_pose': lidar_pose})


        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        ### 2022.10.10 single gt ####
        if self.supervise_single:
            output_dict['ego'].update({
                "label_dict_single" : 
                    {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                     "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                     "targets": torch.cat(targets_single, dim=0)}
            })

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        # transformation is only used in post process (no use.)
        # we all predict boxes in ego coord.
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch,
                                    'transformation_matrix_clean':
                                       transformation_matrix_clean_torch,})

        output_dict['ego'].update({
            "sample_idx": batch[0]['ego']['sample_idx'],
            "cav_id_list": batch[0]['ego']['cav_id_list']
        })

        return output_dict

    def generate_object_center(
        self, cav_contents, reference_lidar_pose
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
            cav_contents, reference_lidar_pose
        )

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

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['params']['lidar_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
