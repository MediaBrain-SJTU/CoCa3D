"""
3D Anchor Generator for Voxel
"""
import numpy as np
import torch

from opencood.data_utils.post_processor.voxel_postprocessor \
    import VoxelPostprocessor
from opencood.utils import box_utils
from opencood.utils import common_utils
from opencood.utils.common_utils import limit_period

class FVoxelRCNNPostprocessor(VoxelPostprocessor):
    def __init__(self, anchor_params, train):
        super(FVoxelRCNNPostprocessor, self).__init__(anchor_params, train)

    def post_process(self, data_dict, output_dict, stage1=False):
        if stage1:
            return self.post_process_stage1(data_dict, output_dict)
        else:
            return self.post_process_stage2(data_dict)

    def post_process_stage1(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        No NMS


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_original_list = []
        pred_box3d_list = []
        pred_box2d_list = []

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            # the transformation matrix to ego space
            if 'transformation_matrix' in cav_content:
                transformation_matrix = cav_content['transformation_matrix']
            else:
                transformation_matrix = torch.from_numpy(np.identity(4)).float().\
                    to(cav_content['anchor_box'].device)


            # (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # prediction result
            preds_dict = output_dict[cav_id]['preds_dict_stage1']

            # preds
            prob = preds_dict['cls_preds']
            prob = torch.sigmoid(prob.permute(0, 2, 3, 1).contiguous())
            reg = preds_dict['box_preds'] 

            # convert regression map back to bounding box
            # (N, W*L*anchor_num, 7)
            # reg: (N, anchor_num*7, H, W), permute inside delta_to_boxes3d

            batch_box3d = self.delta_to_boxes3d(reg, anchor_box) # hwl
            mask = torch.gt(prob, self.params['target_args']['score_threshold'])
            batch_num_box_count = [int(m.sum()) for m in mask]
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            if not self.train:
                assert batch_box3d.shape[0] == 1

            boxes3d = torch.masked_select(batch_box3d.view(-1, 7), mask_reg[0]).view(-1, 7) # hwl. right
            scores = torch.masked_select(prob.view(-1), mask[0])


            dir_labels = torch.max(dir, dim=-1)[1]
            dir_labels = dir_labels[mask]
            # top_labels = torch.zeros([scores.shape[0]], dtype=torch.long).cuda()
            # process per agent
            if scores.shape[0] != 0:
                # adding dir classifier
                dir_offset = self.params['dir_args']['dir_offset']
                num_bins = self.params['dir_args']['num_bins']

                dir_cls_preds = preds_dict['dir_cls_preds'].permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
                period = (2 * np.pi / num_bins) # pi
                dir_rot = limit_period(
                    boxes3d[..., 6] - dir_offset, 0, period
                ) # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

                # filter invalid boxes
                keep_idx = torch.logical_and((boxes3d[:, 3:6] > 1).all(dim=1), (boxes3d[:, 3:6] < 10).all(dim=1))
                idx_start = 0
                count = []
                for i, n in enumerate(batch_num_box_count):
                    count.append(int(keep_idx[idx_start:idx_start+n].sum()))
                batch_num_box_count = count
                boxes3d = boxes3d[keep_idx] # hwl
                scores = scores[keep_idx]


                # if the number of boxes is too huge, this would consume a lot of memory in the second stage
                # therefore, randomly select some boxes if the box number is too big at the beginning of the training
                if len(boxes3d) > 300:
                    keep_idx = torch.multinomial(scores, 300)
                    idx_start = 0
                    count = []
                    for i, n in enumerate(batch_num_box_count):
                        count.append(int(torch.logical_and(keep_idx>=idx_start, keep_idx<idx_start + n).sum()))
                    batch_num_box_count = count
                    boxes3d = boxes3d[keep_idx] 
                    scores = scores[keep_idx]

                pred_box3d_original_list.append(boxes3d.detach()) # hwl
            
            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.params['order'])
                # (N, 8, 3)
                projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)


        if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)

        pred_box3d_original = torch.vstack(pred_box3d_original_list)

        cur_idx = 0
        batch_pred_boxes3d = []
        batch_scores = []
        for n in batch_num_box_count:
            cur_boxes = pred_box3d_tensor[cur_idx:cur_idx+n]
            cur_scores = scores[cur_idx:cur_idx+n]
            # nms
            keep_index = box_utils.nms_rotated(cur_boxes,
                                               cur_scores,
                                               self.params['nms_thresh']
                                               )
            cur_boxes = pred_box3d_original[cur_idx:cur_idx+n][:, [0, 1, 2, 5, 4, 3, 6]] 
            batch_pred_boxes3d.append(cur_boxes[keep_index])
            batch_scores.append(cur_scores[keep_index])
            cur_idx += n
            # print("stage1 cur_boxes:",cur_boxes) # hwl

        return batch_pred_boxes3d, batch_scores

    def post_process_stage2(self, data_dict):
        from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import nms_gpu
        if 'fvoxelrcnn_out' not in data_dict['ego'].keys():
            return None, None
        output_dict = data_dict['ego']['fvoxelrcnn_out']
        label_dict = data_dict['ego']['rcnn_label_dict']
        rcnn_cls = output_dict['rcnn_cls'].sigmoid().view(-1)
        rcnn_reg = output_dict['rcnn_reg'].view(-1, 7)
        rois_anchor = label_dict['rois_anchor'] # rois_anchor is hwl order
        rois = label_dict['rois']
        roi_center = rois[:, 0:3]
        roi_ry = rois[:, 6] % (2 * np.pi)
        boxes_local = box_utils.box_decode(rcnn_reg, rois_anchor)

        detections = common_utils.rotate_points_along_z(
            points=boxes_local.view(-1, 1, boxes_local.shape[-1]), angle=roi_ry.view(-1)
        ).view(-1, boxes_local.shape[-1])
        detections[:, :3] = detections[:, :3] + roi_center
        detections[:, 6] = detections[:, 6] + roi_ry

        mask = nms_gpu(detections, rcnn_cls, thresh=0.01)[0]
        boxes3d = detections[mask][:, [0, 1, 2, 5, 4, 3, 6]] # lwh -> hwl

        projected_boxes3d = None
        if len(boxes3d) != 0:
            # (N, 8, 3)
            boxes3d_corner = \
                box_utils.boxes_to_corners_3d(boxes3d,
                                              order=self.params['order'])
            # (N, 8, 3)
            projected_boxes3d = \
                box_utils.project_box3d(boxes3d_corner,
                                        data_dict['ego']['transformation_matrix'])

        ## Added by Yifan Lu, filter box outside of GT range
        if projected_boxes3d is None:
            return None, None
        scores = scores[mask]
        cav_range = self.params['gt_range']
        mask = box_utils.get_mask_for_boxes_within_range_torch(projected_boxes3d, cav_range)
        projected_boxes3d = projected_boxes3d[mask]
        scores = scores[mask]


        return projected_boxes3d, scores
