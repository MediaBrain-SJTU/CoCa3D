# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from typing import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=20,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--cavnum', type=int, default=5, 
                        help="number of agent in collaboration")
    parser.add_argument('--fix_cavs_box', dest='fcb', action='store_true',
                        help="fix(add) bounding box for cav(s)")
    parser.add_argument('--depth_metric', '-d', action='store_true',
                        help="evaluate depth estimation performance")
    parser.add_argument('--note', default="[newtest_ego_all_gt]", type=str, help="any other thing?")
    parser.set_defaults(fcb=True)
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if "OPV2V" in hypes['test_dir'] else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    ##############################
    # Use ego agent's all gt.
    if "Camera" in hypes['fusion']['core_method']:
        from opencood.hypes_yaml.yaml_utils import load_lift_splat_shoot_params
        # use updated pkl
        if "/valid_data_info.pkl" in hypes['valid_data']:
            hypes['valid_data'] = hypes['valid_data'].replace("valid_data_info.pkl", "updated_valid_data_info.pkl")
            # hypes['valid_data'] = hypes['valid_data'].replace("valid_data_info.pkl", "updated_valid_data_info_15.pkl")
            # hypes['valid_data'] = hypes['valid_data'].replace("valid_data_info.pkl", "updated_valid_data_info_agent15.pkl")
        # use infer dataset
        hypes['fusion']['core_method'] += "_"
        # unlimited comm range
        hypes['comm_range'] = 1e5
        # set detection range
        x_axis_range, y_axis_range = opt.range.split(",")
        x_min, x_max = -eval(x_axis_range), eval(x_axis_range)
        y_min, y_max = -eval(y_axis_range), eval(y_axis_range)
        hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
        hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
        hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                         x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
        hypes['preprocess']['cav_lidar_range'] =  new_cav_range
        hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['gt_range'] = new_cav_range
        hypes['train_params']['max_cav'] = opt.cavnum
        # reload anchor
        hypes = load_lift_splat_shoot_params(hypes)
        print(f"\n\nChange to {hypes['fusion']['core_method']} to use ego's all GT \n \n")
    else:
        x_max = hypes['preprocess']['cav_lidar_range'][3]
        y_max = hypes['preprocess']['cav_lidar_range'][4]
    ##############################

    print('Creating Model')
    model = train_utils.create_model(hypes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    noise_setting = OrderedDict()
    noise_setting['add_noise'] = False
    
    # build dataset for each noise setting
    print('Dataset Building')
    print(f"No Noise Added.")
    hypes.update({"noise_setting": noise_setting})
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                0.5: {'tp': [], 'fp': [], 'gt': 0},
                0.7: {'tp': [], 'fp': [], 'gt': 0}}
    
    if opt.depth_metric:
        depth_stat = []
    
    noise_level = "no_noise_"+opt.fusion_method+ f"[_{x_max}m_{y_max}m]" + f"{opt.cavnum}agent" +opt.note

    object_centers = []
    agent_centers = []
    for i, batch_data in enumerate(data_loader):
        print(f"{noise_level}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']

            cavnum = 0
            # if opt.fix_cavs_box:
            if opt.fcb:
                pred_box_tensor, gt_box_tensor, pred_score, cavnum = inference_utils.fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data)

            if pred_box_tensor is not None:
                pred_box_tensor = pred_box_tensor[:,:4,:2].mean(dim=1)
                if cavnum > 0:
                    agent_centers.append(pred_box_tensor.cpu().numpy()[:cavnum])
                
            if gt_box_tensor is not None:
                gt_box_tensor = gt_box_tensor[:,:4,:2].mean(dim=1)
                object_centers.append(gt_box_tensor.cpu().numpy()) 
                

        torch.cuda.empty_cache()

    object_centers = np.concatenate(object_centers, axis=0)
    agent_centers = np.concatenate(agent_centers, axis=0)

    np.save('object_centers.npy', object_centers)
    np.save('agent_centers.npy', agent_centers)
    

if __name__ == '__main__':
    main()
