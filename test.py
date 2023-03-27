from functools import partial

import numpy as np
from skimage import transform    

VOXEL_SIZE = [0.16, 0.16, 4]
MAX_POINTS_PER_VOXEL = 32
MAX_NUMBER_OF_VOXELS = 16000
point_cloud_range = np.array([0, -39.68, -3, 69.12, 39.68, 1])


def transform_points_to_voxels(data_dict=None, config=None, voxel_generator=None):
    """
    将点云转换为voxel,调用spconv的VoxelGeneratorV2
    """
    try:
        from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
    except:
        from spconv.utils import VoxelGenerator

    voxel_generator = VoxelGenerator(
        voxel_size=VOXEL_SIZE, # [0.16, 0.16, 4]
        point_cloud_range=point_cloud_range, # [0, -39.68, -3, 69.12, 39.68, 1]
        max_num_points=MAX_POINTS_PER_VOXEL, # 32
        max_voxels=MAX_NUMBER_OF_VOXELS # 16000
    )
    grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(VOXEL_SIZE) # 网格数量
    grid_size = np.round(grid_size).astype(np.int64) 
    print("grid_size: ",grid_size)
    voxel_size = VOXEL_SIZE

    # 调用spconv的voxel_generator的generate方法生成体素
    points = data_dict['points']
    voxel_output = voxel_generator.generate(points)
    """
        voxels: (num_voxels, max_points_per_voxel, 3 + C)
        coordinates: (num_voxels, 3) # zyx
        num_points: (num_voxels)
    """
    if isinstance(voxel_output, dict):
        voxels, coordinates, num_points = \
            voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
    else:
        voxels, coordinates, num_points = voxel_output

    if not data_dict['use_lead_xyz']:
        voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

    data_dict['voxels'] = voxels
    data_dict['voxel_coords'] = coordinates
    data_dict['voxel_num_points'] = num_points

    print(voxels)
    print(coordinates)
    print(num_points)
    return data_dict


if __name__=="__main__":
    point_x = np.arange(20)
    point_y = np.arange(20)
    point_z = np.ones(20) / 2
    points = np.stack((point_x, point_y, point_z),axis=1)
    print(points)
    data_dict = {"points":points, "use_lead_xyz":True}
    data_dict = transform_points_to_voxels(data_dict)
