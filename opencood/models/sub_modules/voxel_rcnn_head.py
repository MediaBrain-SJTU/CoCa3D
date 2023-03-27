import opencood.utils.common_utils as common_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils import box_utils
import numpy as np
import torch
import torch.nn as nn

class RoIHead(nn.Module):
    """
    1. generate label for rcnn head
    2. box refinement
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        c_out = model_cfg['c_out']
        self.code_size = 7
        self.num_class = model_cfg['num_class']
        self.grid_size = model_cfg['grid_size']

        ####### these are for RoI head #######
        pre_channel = self.grid_size * self.grid_size * self.grid_size * c_out
        shared_fc_list = []
        for k in range(0, self.model_cfg['shared_fc'].__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg['shared_fc'][k], bias=False),
                nn.BatchNorm1d(self.model_cfg['shared_fc'][k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg['shared_fc'][k]

            if k != self.model_cfg['shared_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)


        cls_fc_list = []
        for k in range(0, self.model_cfg['cls_fc'].__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg['cls_fc'][k], bias=False),
                nn.BatchNorm1d(self.model_cfg['cls_fc'][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg['cls_fc'][k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg['reg_fc'].__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg['reg_fc'][k], bias=False),
                nn.BatchNorm1d(self.model_cfg['reg_fc'][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg['reg_fc'][k]

            if k != self.model_cfg['reg_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.code_size * self.num_class, bias=True)

    def assign_targets(self, batch_dict):
        batch_dict['rcnn_label_dict'] = {
            'rois': [],
            'gt_of_rois': [],
            'gt_of_rois_src': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
            'rois_anchor': [],
            'record_len': []
        }
        pred_boxes = batch_dict['boxes_fused']  # lwh
        gt_boxes = [b[m][:, [0, 1, 2, 5, 4, 3, 6]].float() for b, m in
                    zip(batch_dict['object_bbx_center'],
                        batch_dict['object_bbx_mask'].bool())]  # lwh order
        for rois, gts in zip(pred_boxes, gt_boxes): # each frame
            # print("rois:",rois)  # lwh
            # print('gts',gts) # lwh

            ious = boxes_iou3d_gpu(rois, gts)
            max_ious, gt_inds = ious.max(dim=1)
            gt_of_rois = gts[gt_inds]
            rcnn_labels = (max_ious > 0.3).float()
            mask = torch.logical_not(rcnn_labels.bool())

            # set negative samples back to rois, no correction in stage2 for them
            gt_of_rois[mask] = rois[mask]
            gt_of_rois_src = gt_of_rois.clone().detach()

            # canoical transformation
            roi_center = rois[:, 0:3]

            roi_ry = rois[:, 6] % (2 * np.pi)
            gt_of_rois[:, 0:3] = gt_of_rois[:, 0:3] - roi_center
            gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]),
                angle=-roi_ry.view(-1)
            ).view(-1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = (gt_of_rois[:, 6] + (
                    torch.div(torch.abs(gt_of_rois[:, 6].min()),
                              (2 * np.pi), rounding_mode='trunc')
                    + 1) * 2 * np.pi) % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (
                    heading_label < np.pi * 1.5)

            # (0 ~ pi/2, 3pi/2 ~ 2pi)
            heading_label[opposite_flag] = (heading_label[
                                                opposite_flag] + np.pi) % (
                                                   2 * np.pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[
                                      flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2,
                                        max=np.pi / 2)
            gt_of_rois[:, 6] = heading_label

            # generate regression target
            rois_anchor = rois.clone().detach().view(-1, self.code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0

            reg_targets = box_utils.box_encode(
                gt_of_rois.view(-1, self.code_size), rois_anchor
            )

            batch_dict['rcnn_label_dict']['rois'].append(rois)
            batch_dict['rcnn_label_dict']['gt_of_rois'].append(gt_of_rois)
            batch_dict['rcnn_label_dict']['gt_of_rois_src'].append(
                gt_of_rois_src)
            batch_dict['rcnn_label_dict']['cls_tgt'].append(rcnn_labels)
            batch_dict['rcnn_label_dict']['reg_tgt'].append(reg_targets)
            batch_dict['rcnn_label_dict']['iou_tgt'].append(max_ious)
            batch_dict['rcnn_label_dict']['rois_anchor'].append(rois_anchor)
            batch_dict['rcnn_label_dict']['record_len'].append(rois.shape[0])
            

        # cat list to tensor
        for k, v in batch_dict['rcnn_label_dict'].items():
            if k == 'record_len':
                continue
            batch_dict['rcnn_label_dict'][k] = torch.cat(v, dim=0)

        return batch_dict

    def forward(self, batch_dict):
        pooled_features = batch_dict['pooled_features'] # # (BxN, 6x6x6, C)
        
        # Box Refinement (batch * N, 6x6x6, 96) --> (batch * N, 6x6x6x96)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        shared_features = self.shared_fc_layer(pooled_features)
        # (256, 1)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        # (256, 7)
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        batch_dict['fvoxelrcnn_out'] = {
            'rcnn_cls': rcnn_cls,
            'rcnn_reg': rcnn_reg,
        }
        return batch_dict