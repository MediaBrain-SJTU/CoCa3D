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
from PIL import Image
import pickle as pkl
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import camera_basedataset
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
    coord_3d_to_2d,
)
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import read_json
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum
)


class CameraLateFusionDataset(camera_basedataset.CameraBaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(CameraLateFusionDataset, self).__init__(params, visualize, train)
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.anchor_box = self.post_processor.generate_anchor_box()
        self.anchor_box_torch = torch.from_numpy(self.anchor_box)
        
        if self.preload and self.preload_worker_num:
            self.retrieve_all_base_data_mp()
        elif self.preload:
            self.retrieve_all_base_data()


    def __getitem__(self, idx):
        if self.preload:
            base_data_dict = self.all_base_data[idx]
        else:
            base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            # reformat_data_dict = self.get_item_train(base_data_dict)
            reformat_data_dict = self.get_item_test(base_data_dict, idx)

        return reformat_data_dict

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
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose, visibility_map
        )

    def get_item_single_car_camera(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.


        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
            including 'params', 'camera_data'

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # filter lidar
        if self.visualize:
            lidar_np = selected_cav_base["lidar_np"]
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_points_by_range(
                lidar_np, self.params["preprocess"]["cav_lidar_range"]
            )
            # remove points that hit ego vehicle
            lidar_np = mask_ego_points(lidar_np)
            selected_cav_processed.update({"origin_lidar": lidar_np})

        # single label dict
        # generate the bounding box(n, 7) under the cav's space
        selected_cav_base["bev_visibility.png"] = cv2.cvtColor(selected_cav_base["bev_visibility.png"], cv2.COLOR_BGR2GRAY)
        visibility_map = np.asarray(selected_cav_base["bev_visibility.png"])
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], selected_cav_base["params"]["lidar_pose_clean"], visibility_map
        )

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
            }
        )

        # generate targets label
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
        )
        selected_cav_processed.update({"label_dict": label_dict})


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
                np.float32)
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
                                camera_to_lidar) 
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
            if self.use_fg_mask:
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
                    "imgs": torch.stack(imgs), # [N, 3or4, H, W]
                    "intrins": torch.stack(intrins),
                    "rots": torch.stack(rots),
                    "trans": torch.stack(trans),
                    "post_rots": torch.stack(post_rots),
                    "post_trans": torch.stack(post_trans),
                }
            }
        )


        return selected_cav_processed

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()
        base_data_dict = add_noise_data_dict(
            base_data_dict, self.params["noise_setting"]
        )
        # during training, we return a random cav's data
        # only one vehicle is in processed_data_dict
        if not self.visualize:
            selected_cav_id, selected_cav_base = random.choice(
                list(base_data_dict.items())
            )
        else:
            selected_cav_id, selected_cav_base = list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car_camera(selected_cav_base)
        processed_data_dict.update({"ego": selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict, idx):
        """
        processed_data_dict.keys() = ['ego', "650", "659", ...]
        """
        base_data_dict = add_noise_data_dict(
            base_data_dict, self.params["noise_setting"]
        )

        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []
        cav_id_list = []
        lidar_pose_list = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id = cav_id
                ego_lidar_pose = cav_content["params"]["lidar_pose"]
                ego_lidar_pose_clean = cav_content["params"]["lidar_pose_clean"]
                break

        if ego_id == -1:
            ego_id = list(base_data_dict.keys())[0]
        # assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt(
                (selected_cav_base["params"]["lidar_pose"][0] - ego_lidar_pose[0]) ** 2
                + (selected_cav_base["params"]["lidar_pose"][1] - ego_lidar_pose[1])
                ** 2
            )
            if distance > self.params["comm_range"]:
                continue
            cav_id_list.append(cav_id)
            lidar_pose_list.append(selected_cav_base["params"]["lidar_pose"])

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]
            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base["params"]["lidar_pose"]
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
            cav_lidar_pose_clean = selected_cav_base["params"]["lidar_pose_clean"]
            transformation_matrix_clean = x1_to_x2(
                cav_lidar_pose_clean, ego_lidar_pose_clean
            )

            selected_cav_processed = self.get_item_single_car_camera(selected_cav_base)
            selected_cav_processed.update(
                {
                    "transformation_matrix": transformation_matrix,
                    "transformation_matrix_clean": transformation_matrix_clean,
                }
            )
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict

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
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            label_dict_list.append(ego_dict["label_dict"])

            if self.visualize:
                origin_lidar.append(ego_dict["origin_lidar"])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "anchor_box": self.anchor_box_torch,
                "label_dict": label_torch_dict,
            }
        )

        # collate ego camera information
        imgs_batch = []
        rots_batch = []
        trans_batch = []
        intrins_batch = []
        post_trans_batch = []
        post_rots_batch = []
        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]["image_inputs"]
            imgs_batch.append(ego_dict["imgs"])
            rots_batch.append(ego_dict["rots"])
            trans_batch.append(ego_dict["trans"])
            intrins_batch.append(ego_dict["intrins"])
            post_trans_batch.append(ego_dict["post_trans"])
            post_rots_batch.append(ego_dict["post_rots"])

        output_dict["ego"].update(
            {
                "imgs": torch.stack(imgs_batch),  # [B, N, C, H, W]
                "rots": torch.stack(rots_batch),
                "trans": torch.stack(trans_batch),
                "intrins": torch.stack(intrins_batch),
                "post_trans": torch.stack(post_trans_batch),
                "post_rots": torch.stack(post_rots_batch),
            }
        )

        return output_dict

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
            object_bbx_center = torch.from_numpy(
                np.array([cav_content["object_bbx_center"]])
            )
            object_bbx_mask = torch.from_numpy(
                np.array([cav_content["object_bbx_mask"]])
            )
            object_ids = cav_content["object_ids"]

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            output_dict[cav_id].update(
                {
                    "anchor_box": self.anchor_box_torch
                }
            )
            
            if self.visualize:
                transformation_matrix = cav_content["transformation_matrix"]
                origin_lidar = [cav_content["origin_lidar"]]

                if (self.params["only_vis_ego"] is False) or (cav_id == "ego"):
                    projected_lidar = copy.deepcopy(cav_content["origin_lidar"])
                    projected_lidar[:, :3] = box_utils.project_points_by_matrix_torch(
                        projected_lidar[:, :3], transformation_matrix
                    )
                    projected_lidar_list.append(projected_lidar)

            # label dictionary
            label_torch_dict = self.post_processor.collate_batch(
                [cav_content["label_dict"]]
            )

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = torch.from_numpy(
                np.array(cav_content["transformation_matrix"])
            ).float()

            # late fusion training, no noise
            transformation_matrix_clean_torch = transformation_matrix_torch

            imgs_batch = [cav_content["image_inputs"]["imgs"]]
            rots_batch = [cav_content["image_inputs"]["rots"]]
            trans_batch = [cav_content["image_inputs"]["trans"]]
            intrins_batch = [cav_content["image_inputs"]["intrins"]]
            post_trans_batch = [cav_content["image_inputs"]["post_trans"]]
            post_rots_batch = [cav_content["image_inputs"]["post_rots"]]

            output_dict[cav_id].update(
                {
                    "imgs": torch.stack(imgs_batch),
                    "rots": torch.stack(rots_batch),
                    "trans": torch.stack(trans_batch),
                    "intrins": torch.stack(intrins_batch),
                    "post_trans": torch.stack(post_trans_batch),
                    "post_rots": torch.stack(post_rots_batch),
                }
            )

            output_dict[cav_id].update(
                {
                    "object_bbx_center": object_bbx_center,
                    "object_bbx_mask": object_bbx_mask,
                    "label_dict": label_torch_dict,
                    "object_ids": object_ids,
                    "transformation_matrix": transformation_matrix_torch,
                    "transformation_matrix_clean": transformation_matrix_clean_torch,
                }
            )

        if self.visualize:
            projected_lidar_stack = [torch.from_numpy(np.vstack(projected_lidar_list))]
            output_dict["ego"].update({"origin_lidar": projected_lidar_stack})
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
        pred_box_tensor, pred_score = self.post_processor.post_process(
            data_dict, output_dict
        )
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_no_fusion(
        self, data_dict, output_dict_ego, return_uncertainty=False
    ):
        data_dict_ego = OrderedDict()
        data_dict_ego["ego"] = data_dict["ego"]
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        if return_uncertainty:
            pred_box_tensor, pred_score, uncertainty = self.post_processor.post_process(
                data_dict_ego, output_dict_ego, return_uncertainty=True
            )
            return pred_box_tensor, pred_score, gt_box_tensor, uncertainty
        else:
            pred_box_tensor, pred_score = self.post_processor.post_process(
                data_dict_ego, output_dict_ego
            )
            return pred_box_tensor, pred_score, gt_box_tensor



