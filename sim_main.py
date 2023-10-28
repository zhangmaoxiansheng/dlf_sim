import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

from PIL import Image
from tqdm import tqdm

import open3d as o3d
import random
import copy
from glob import glob

from ply_io import ply_io
from bin2txt import bin2txt

from cam import camera,get_camera,red_check
from cam import camera_group
from scene import scene_generator

def create_aabb_voxel(aabb_points,voxel_size=5):
    # 计算AABB的最小和最大顶点坐标
    min_x = min([point[0] for point in aabb_points])
    min_y = min([point[1] for point in aabb_points])
    min_z = min([point[2] for point in aabb_points])

    max_x = max([point[0] for point in aabb_points])
    max_y = max([point[1] for point in aabb_points])
    max_z = max([point[2] for point in aabb_points])

    # 计算AABB框的尺寸
    aabb_width = max_x - min_x
    aabb_height = max_y - min_y
    aabb_depth = max_z - min_z

    # 计算每个轴上的体素数量
    num_x_voxels = int(aabb_width / voxel_size)
    num_y_voxels = int(aabb_height / voxel_size)
    num_z_voxels = int(aabb_depth / voxel_size)

    # 生成体素中心点坐标
    voxel_centers = []
    for i in range(num_x_voxels):
        for j in range(num_y_voxels):
            for k in range(num_z_voxels):
                x = min_x + i * voxel_size + voxel_size / 2
                y = min_y + j * voxel_size + voxel_size / 2
                z = min_z + k * voxel_size + voxel_size / 2
                voxel_centers.append([x, y, z])
    voxel_centers = np.asarray(voxel_centers)
    # 将体素中心点坐标转换为Open3D的体素格式
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)
    #point_cloud.paint_uniform_color([0.5,0.5,0.5])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
    return voxel_centers, voxel_grid

def check_voxel(voxel_grid, queries, voxel_size=1):
    #point cloud is the center of the voxel
    output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    colors = np.array([[1, 0, 0] if i else [0.83,0.83,0.83] for i in output])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(queries)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud, output

def get_cam_set(txt_dir,scale = 1):
    if not os.path.exists(os.path.join(txt_dir,'images.txt')):
        bin2txt(txt_dir) 
    cameras_all = get_camera(txt_dir)

    cam_group = {}

    for cam in cameras_all:
        group_name = cam.name.split('/')[0]
        group_name = group_name.replace('1020-','')
        if group_name not in cam_group:
            cam_group[group_name] = []
        cam_group[group_name].append(cam)

    cam_set = [camera_group(cam_group[f'{i}']) for i in range(0,len(cam_group))]
    return cam_set


def generate_scene():
    txt_dir = './1020_colmap/dense/sparse'    

    cam_set = get_cam_set(txt_dir,scale=14.2887)
    
    cameras = cam_set[5]

    target_point = [-350.33,101.15799,-356.793213]
    target_camera_point = o3d.geometry.PointCloud()
    target_camera_point.points = o3d.utility.Vector3dVector([np.asarray(target_point)])
    target_camera_point.paint_uniform_color([0, 1, 0])
    cameras.transform_to(target_point,[100,95,0])
    cameras.transform_to(target_point,[180,0,0])
    cameras.transform_to(target_point,[0,0,179.7])
    cameras.transform_to(target_point,[5,0,0])

    cam_points, frustums = cameras.get_view_frustums()

    cube_names = ["scube-a.obj","scube-red-l.obj","scube4.obj","scube-red-s.obj"]
    cube_names = [os.path.join('./meshes/small_cubes',i) for i in cube_names]

    base = './meshes/base.obj'
    pos_list = np.asarray([[-90,166,-454.656],[-91,61,-454.656],[-49,-29,-454.626],[-57.5818,244.043,-454.656],[-102,314.316,-454.656],[9.15,-110.085,-454.656],[-96,238,-454.656],[-15,-160,-454.656],[17,212,454.656]])
    #only xy plane
    static_objs = ['./meshes/walls.obj','./meshes/jianshan_nobase.obj']
    scene = scene_generator(base,cube_names,pos_list,static_objs)
    scene.generate('walls',[20.6179,49.535126,-326.4666])
    scene.generate('jianshan_nobase',[55.894,100.657,-395.297])
    for i in range(10):
        index = i % (len(pos_list)-1)
        if i < 2:
            scene.random_generate(key = 'scube-red-l',pos_index = index)
        elif i < 5:
            scene.random_generate(key = 'scube-red-s',pos_index = index)
        else:
            scene.random_generate(pos_index = index)

    all_vis = [target_camera_point] + cam_points + frustums + scene.current_scene
    
    o3d.visualization.draw_geometries(all_vis)
    cameras.save_camera_pos('./')
    rgbs,deps = cameras.capture(scene.current_scene,save_dir = './',show=False,with_depth=True)

    scene.save_all("./meshes/test_scene/test_mesh.obj")

def capture_test(group_id,vis = False):
    txt_dir = './1020_colmap/dense/sparse'    
    cam_set = get_cam_set(txt_dir,scale=14.2887)
    
    cameras = cam_set[group_id]

    target_point = [-350.33,101.15799,-356.793213]
    target_camera_point = o3d.geometry.PointCloud()
    target_camera_point.points = o3d.utility.Vector3dVector([np.asarray(target_point)])
    target_camera_point.paint_uniform_color([0, 1, 0])
    cameras.transform_to(target_point,[100,95,0])
    cameras.transform_to(target_point,[180,0,0])
    cameras.transform_to(target_point,[0,0,179.7])
    cameras.transform_to(target_point,[5,0,0])

    cam_points, frustums = cameras.get_view_frustums()

    mesh = o3d.io.read_triangle_mesh("./meshes/test_scene/test_mesh.obj", True)

    if vis:
        all_vis = [mesh,target_camera_point] + cam_points + frustums
        o3d.visualization.draw_geometries(all_vis)

    cameras.save_camera_pos('./')
    save_dir = f'./{group_id}'
    rgbs,deps = cameras.capture([mesh],save_dir = save_dir,show=False,with_depth=True)

def voxel_check_test_sim_depth(cameras,pcd_dir,vis = True):
    if not os.path.exists(os.path.join(pcd_dir,'points3D.ply')):
        ply_io(pcd_dir)

    pcd_dir = os.path.join(pcd_dir,'points3D.ply')

    cam_points = []
    frustums = []
    
    #cameras = camera_group(cameras_all)
    
    cam_points, frustums = cameras.get_view_frustums(line_radius=0.05)
    
    
    aabb,aabb_p = cameras.get_aabb()
    center,aabb_voxel_grid = create_aabb_voxel(aabb_p,voxel_size=1)

    pcd = o3d.io.read_point_cloud(pcd_dir)
    pcd = pcd.crop(aabb)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.8)
    rgbs,depths = cameras.capture([pcd],save_dir = './sim_test', show=False, with_depth=True, save_in_cam=True)

    occu_pcd, occu_list = check_voxel(voxel_grid,center)
    overlap_pcd, filter_pcd, score_array = cameras.check_visible_overlap(center, voxel_grid, occu_list)
    score = np.sum(score_array)
    print(f'score {score}')
    overlap_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(overlap_pcd,voxel_size=0.15)
    filter_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(filter_pcd,voxel_size=0.2)
    
    all_vis = [voxel_grid, aabb,overlap_voxel_grid, filter_voxel_grid] + cam_points + frustums
    #all_vis = [voxel_grid, aabb,overlap_voxel_grid] + cam_points + frustums
    #all_vis = [voxel_grid, aabb,overlap_voxel_grid, filter_voxel_grid, cam_points[0]] +  frustums[:16]
    if vis:
        o3d.visualization.draw_geometries(all_vis)
    return score, voxel_grid, filter_pcd, overlap_voxel_grid

def voxel_check_test_sim_voxel(cameras,voxel_grid,filter_pcd,vis = False):
    filter_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(filter_pcd,voxel_size=0.3)

    cam_points = []
    frustums = []
    
    # cameras = camera_group(cam_group[f'{i}'])
    #cameras = camera_group(cameras_all)
    
    cam_points, frustums = cameras.get_view_frustums(line_radius=0.05)
    
    
    aabb,aabb_p = cameras.get_aabb()
    center,aabb_voxel_grid = create_aabb_voxel(aabb_p,voxel_size=1)

    #pcd = o3d.io.read_point_cloud(pcd_dir)
    
    #pcd = pcd.crop(aabb)
    #voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.8)
    #rgbs,depths = cameras.capture([pcd],save_dir = './sim_test', show=False, with_depth=True, save_in_cam=True)

    occu_pcd, occu_list1 = check_voxel(voxel_grid,center)
    occu_pcd, occu_list2 = check_voxel(filter_voxel_grid,center)
    occu_list = [i or j for i,j in zip(occu_list1,occu_list2)]
    overlap_pcd, filter_pcd, score_array = cameras.check_visible_overlap(center, voxel_grid, occu_list)
    score = np.sum(score_array)
    print(f'score {score}')
    overlap_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(overlap_pcd,voxel_size=0.15)
    #filter_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(filter_pcd,voxel_size=0.3)
    
    all_vis = [voxel_grid, aabb, overlap_voxel_grid] + cam_points + frustums
    if vis:
        o3d.visualization.draw_geometries(all_vis)
    return score

def voxel_check_test_sim():
    txt_dir = './0/sparse/0'
    pcd_dir = './0/sparse/0/'
    if not os.path.exists(os.path.join(txt_dir,'images.txt')):
        bin2txt(txt_dir)
    
    cameras_all = get_camera(txt_dir)
    cameras0 = camera_group(cameras_all,'3')
    
    txt_dir = './1020_colmap/dense/sparse'   
    cam_set = get_cam_set(txt_dir)
    scale = cameras0.cal_neighbor_dist() / cam_set[0].cal_neighbor_dist()
    for cg in cam_set:
        cg.adjust_scale(scale)
        cg.reset_pose_all(cameras0.center_cam.pose)

    score, voxel_grid, filter_pcd, overlap_voxel_grid = voxel_check_test_sim_depth(cameras0,pcd_dir,vis=True)
    
    for i in range(0,len(cam_set)):
        cg = cam_set[i]
        voxel_check_test_sim_voxel(cg, voxel_grid,filter_pcd,True)
    
if __name__ == '__main__':
    #capture_test(group_id=5,vis=True)
    voxel_check_test_sim()
    
    #for i in range(5):
        #capture_test(group_id=i)
    #red_check('./sim_image_0.png')
    
    












# txt_dir = './26_all/sparse/0'
    # pcd_dir = './26_all/sparse/0/points3D.ply'

    # cameras_all = get_camera(txt_dir,22)

    # cam_points = []
    # frustums = []
    # cam_group = {}
    # neighbor_points = []

    # for cam in cameras_all:
    #     group_name = cam.name.split('_')[0]
    #     if group_name not in cam_group:
    #         cam_group[group_name] = []
    #     cam_group[group_name].append(cam)
    
    # i=0
    # cameras = camera_group(cam_group[f'{i}'])

    # target_point = [-199,80,-50]
    # #target_point = [-120,80,-50]
    # rotation_angle = [-65,90,0]
    # target_camera_point = o3d.geometry.PointCloud()
    # target_camera_point.points = o3d.utility.Vector3dVector([np.asarray(target_point)])
    # target_camera_point.paint_uniform_color([0, 1, 0]) 
    # cameras.transform_to(target_point,rotation_angle)

    # rotation_angle = [35,0,5]
    # cameras.transform_to(target_point,rotation_angle)


    # cam_points, frustums = cameras.get_view_frustums()

    # textured_mesh = o3d.io.read_triangle_mesh("./meshes/large_cubes_nobase-5.obj", True)
    # all_vis = [textured_mesh,target_camera_point] + cam_points + frustums
    # #all_vis = [target_camera_point] + frustums
    # o3d.visualization.draw_geometries(all_vis)
    # cameras.save_camera_pos('./')
    # cameras.capture([textured_mesh],show=True)