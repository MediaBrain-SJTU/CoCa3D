# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from torch.utils.data import DataLoader, Subset
from opencood.data_utils import datasets
import torch
from opencood.tools import train_utils, inference_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils, simple_vis
from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR
import numpy as np

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/visualization_dair.yaml'))
    output_path = "/GPFS/rhome/yifanlu/OpenCOOD/vis_result/dair-v2x-c_after"

    opencda_dataset = IntermediateFusionDatasetDAIR(params, visualize=True,
                                            train=True)
    dataset_len = len(opencda_dataset)
    subset_idx = range(10, dataset_len)
    subset_dataset = Subset(opencda_dataset, subset_idx)
    
    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=2,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    vis_gt_box = True
    vis_pred_box = False
    hypes = params

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, batch_data in enumerate(data_loader):
        print(i)
        batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data)

        # vis_save_path = os.path.join(output_path, '3d_%05d.png' % i)
        # simple_vis.visualize(None,
        #                     gt_box_tensor,
        #                     batch_data['ego']['origin_lidar'][0],
        #                     hypes['postprocess']['gt_range'],
        #                     vis_save_path,
        #                     method='3d',
        #                     vis_gt_box = vis_gt_box,
        #                     vis_pred_box = vis_pred_box,
        #                     left_hand=False)
            
        vis_save_path = os.path.join(output_path, 'bev_%05d.png' % i)
        simple_vis.visualize(None,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='bev',
                            vis_gt_box = vis_gt_box,
                            vis_pred_box = vis_pred_box,
                            left_hand=False)