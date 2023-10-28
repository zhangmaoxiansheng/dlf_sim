import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt


import open3d as o3d
import random
import copy



class scene_generator:
    def __init__(self, base, obj_names, seed_pos_list, static_obj_names = [], random_range=180,limit_range = 6) -> None:
        self.objs = {}
        self.static_objs = {}
        self.current_scene = []
        for i in obj_names:
            obj = o3d.io.read_triangle_mesh(i,True)
            key = os.path.basename(i).split('.')[0]
            self.objs[key] = obj

        for i in static_obj_names:
            obj = o3d.io.read_triangle_mesh(i,True)
            key = os.path.basename(i).split('.')[0]
            self.static_objs[key] = obj

        self.pos_list = seed_pos_list[:,:2]
        self.base_z = seed_pos_list[0,2]

        self.random_range = random_range
        self.limited_range = limit_range
        self.limited_pos = []
        self.base = o3d.io.read_triangle_mesh(base,True)
        self.current_scene = [self.base]
        self.init_occupancy_and_condidate()
        

    def init_occupancy_and_condidate(self):
        base_obb_points = self.base.get_oriented_bounding_box().get_box_points()
        base_obb_points = np.asarray(base_obb_points)
        self.base_min = np.asarray([round(np.min(base_obb_points[:,0])),round(np.min(base_obb_points[:,1]))])
        self.base_max = np.asarray([round(np.max(base_obb_points[:,0])),round(np.max(base_obb_points[:,1]))])
        self.occupancy_plane = np.zeros(self.base_max - self.base_min)
        self.activated_plane = 1 - self.occupancy_plane
        self.origin_activate_planes = []#to choose random pos
        for pos in self.pos_list:
            x,y = pos
            x -= self.base_min[0]
            y -= self.base_min[1]
            self.activated_plane[round(x-self.random_range//2):round(x+self.random_range//2), round(y-self.random_range//2):round(y+self.random_range//2)] = 0
            o_activate_planes = 1 - self.occupancy_plane
            o_activate_planes[round(x-self.random_range//2):round(x+self.random_range//2), round(y-self.random_range//2):round(y+self.random_range//2)] = 0
            self.origin_activate_planes.append(o_activate_planes)
        self.candidate_plane = self.occupancy_plane + self.activated_plane
        # plt.imshow(self.candidate_plane)
        # plt.show()
    
    def update_random_range(self,n):
        self.random_range = n

    def update_limit_range(self,n):
        self.limited_range = n

    def update_seed_pos(self, seed_pos_list):
        for pos in seed_pos_list:
            x,y = pos
            x -= self.base_min[0]
            y -= self.base_min[1]
            self.activated_plane[x-self.random_range//2:x+self.random_range//2, y-self.random_range//2:y+self.random_range//2] = 0
        self.pos_list += seed_pos_list

    def update_occupancy_plane(self, pos_xy, obj):
        if pos_xy.shape[0] == 3:
            pos_xy = pos_xy[:2]
        obj_obb_points = obj.get_oriented_bounding_box().get_box_points()
        obj_obb_points = np.asarray(obj_obb_points)
        obj_min = np.asarray([round(np.min(obj_obb_points[:,0])), round(np.min(obj_obb_points[:,1]))])
        obj_max = np.asarray([round(np.max(obj_obb_points[:,0])),round(np.max(obj_obb_points[:,1]))])
        obj_size=  (obj_max - obj_min)*1.5 + self.limited_range
        obj_min = pos_xy - (obj_size // 2) - self.base_min
        obj_max = pos_xy + (obj_size // 2) - self.base_min
        self.occupancy_plane[int(obj_min[0]):int(obj_max[0]),int(obj_min[1]):int(obj_max[1])] = 1
        self.candidate_plane = self.occupancy_plane + self.activated_plane
        self.check_capacity()
    
    def check_capacity(self, thres = 0.7):
        non_occupy_count = np.count_nonzero(self.candidate_plane == 0)
        all_capacity = 0
        for i in self.pos_list:
            all_capacity =all_capacity + self.random_range*self.random_range

        non_occupy_rate = non_occupy_count / all_capacity
        occupy_rate = 1 - non_occupy_rate
        if occupy_rate > thres:
            print(f'warning: insufficient capacity!! now is {occupy_rate}') 
        return occupy_rate

    def random_generate(self,key = None, pos_index = None):
        if key is None:
            key = random.choice(list(self.objs.keys()))
        obj = self.objs[key]
        if pos_index is not None:
            candidate_pos_now = self.candidate_plane + self.origin_activate_planes[pos_index]
        else:
            candidate_pos_now = self.candidate_plane
        # plt.imshow(candidate_pos_now)
        # plt.show()
        candidate_pos = np.where(candidate_pos_now == 0)
        num_elements =  candidate_pos[0].size
        index = np.random.randint(0, num_elements)
        
        pos = np.asarray([candidate_pos[0][index], candidate_pos[1][index]])
        pos += self.base_min 
        
        pos = np.append(pos,self.base_z)
        new_obj = self.generate_cube_at(obj, pos)
        # if self.test_collision(new_obj):
        #     print('collision!')
        #     new_obj = False
        # else:
        print(f'generate {key} at {pos}')
        self.update_occupancy_plane(pos,obj)
        self.current_scene.append(new_obj)
        return new_obj
    
    def generate(self,key,pos):
        if key in self.objs:
            obj = self.objs[key]
        else:
            obj = self.static_objs[key]
        pos = np.asarray(pos)
        #self.update_occupancy_plane(pos[:2],obj)
        if pos.shape[0] == 2:
            pos = np.append(pos, self.base_z)
        new_obj = self.generate_cube_at(obj, pos)
        # if self.test_collision(new_obj):
        #     print('collision!')
        #     return False
        print(f'generate {key} at {pos}')
        self.current_scene.append(new_obj)
        return new_obj
        
    @staticmethod
    def generate_cube_at(cube,pos):
        cube_center = cube.get_center()
        pos = np.asarray(pos)
        translate_vec = pos - cube_center
        new_cube = copy.deepcopy(cube)
        new_cube.translate(translate_vec)
        return new_cube

    def test_collision(self,obj):
        if len(self.current_scene) < 2:
            return False
        for curr_obj in self.current_scene[1:]:
            flag = obj.is_intersecting(curr_obj)
            if flag:
                return True
        return False

    def save_all(self,file_name):
        meshes = self.base
        for c in self.current_scene[1:]:
            meshes += c
        o3d.io.write_triangle_mesh(file_name, meshes ,write_vertex_normals=False, write_vertex_colors=True,write_triangle_uvs=True,print_progress=True)