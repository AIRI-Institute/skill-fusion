import numpy as np
# from numba import njit
import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-1]))
import depth_utils as du
import time
import skimage.measure

from habitat.core.utils import try_cv2_import
from collections import deque

cv2 = try_cv2_import()
from scipy.signal import convolve2d

class MapBuilder(object):

    def __init__(self, params):
        self.params = params

        frame_width = params['frame_width']
        frame_height = params['frame_height']

        self.frame_semantic_width = params['frame_semantic_width']
        self.frame_semantic_height = params['frame_semantic_height']

        fov = params['fov']

        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)

        self.semantic_camera_matrix = du.get_camera_matrix(
            self.frame_semantic_width,
            self.frame_semantic_height,
            fov)

        self.vision_range = params['vision_range']
        self.vision_range_semantic = params['semantic_vision_range']
        print('SEMANTIC VISION RANGE:', self.vision_range_semantic)
        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution']
        self.resolution_semantic = params['resolution']# * n
        self.map_cells = self.map_size_cm // self.resolution
        print('Map cells:', self.map_cells)
        self.obstacle_inflation = params['obstacle_inflation']
        self.semantic_inflation = params['semantic_inflation']

        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']

        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']
        self.classes_number = params['classes_number']
        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)
        self.semantic_map = np.zeros((self.classes_number,
                                      self.map_size_cm // self.resolution_semantic,
                                      self.map_size_cm // self.resolution_semantic,
                                      ), dtype=np.float32)
        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        self.goal_threshold = params['goal_threshold']
        self.goal_decay = params['goal_decay']
        
        return
    
    def set_class_number(self, class_number):
        print("SET CLASS NUMBER", class_number)
        self.classes_number = class_number
        
    def fill_by_bfs(self, geocentric_map, current_pose):

        def normalize_angle(angle):
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            return angle

        def check_borders(x, y, current_pose, visited, vision_range):
            HFOV = 79
            max_deviation = HFOV / (180 / np.pi) / 2
            if x < 0 or x >= self.map_cells or y < 0 or y >= self.map_cells:
                return False
            if visited[x, y] == 1:
                return False
            x_start, y_start, angle = current_pose
            x_cell = round(x_start) / self.resolution
            y_cell = round(y_start) / self.resolution
            if abs(x_cell - x) > vision_range or abs(y_cell - y) > vision_range:
                return False
            direction_to_point = np.arctan2(y - y_cell, x - x_cell)
            if (abs(x_cell - x) > 1 or abs(y_cell - y) > 1) and abs(normalize_angle(direction_to_point - angle)) > max_deviation:
                return False
            return True


        x_start, y_start, angle = current_pose
        x_cell = round(x_start / self.resolution)
        y_cell = round(y_start / self.resolution)
        if x_cell < 0 or x_cell >= self.map_cells:
            return geocentric_map
        if y_cell < 0 or y_cell >= self.map_cells:
            return geocentric_map
        dq = deque()
        dq.append((x_cell, y_cell))
        visited = np.zeros((geocentric_map.shape[0], geocentric_map.shape[1]))
        while len(dq) > 0:
            x, y = dq.popleft()
            #print(x, y)
            if not visited[x, y]:
                if geocentric_map[y, x].sum() == 0:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            if check_borders(x + dx, y + dy, current_pose, visited, self.vision_range):
                                dq.append((x + dx, y + dy))
                    geocentric_map[y, x, 0] = 1
                else:
                    return geocentric_map
            visited[x, y] = 1
        return geocentric_map
        
        
    def get_geocentric_flat(self, depth, camera_matrix, vision_range, resolution, map_shape, current_pose):
        t1 = time.time()
        point_cloud = du.get_point_cloud_from_z(depth, camera_matrix, scale=self.du_scale)
        t2 = time.time()
        #print('Time to get point cloud:', t2 - t1)

        
        shift_loc = [vision_range * resolution // 2, 0, np.pi / 2.0]
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)
        agent_view_centered = du.transform_pose(agent_view, shift_loc)
        agent_view_flat = du.bin_points(
            agent_view_centered,
            vision_range,
            self.z_bins,
            resolution)
        agent_view_cropped = agent_view_flat[:, :, 1]
        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0
        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0
        t3 = time.time()
        #print('Time to get agent view:', t3 - t2)

        geocentric_pc = du.transform_pose(agent_view, current_pose)
        geocentric_flat = du.bin_points(
            geocentric_pc,
            map_shape,
            self.z_bins,
            resolution)
        t4 = time.time()
        #print('Time to bin points:', t4 - t3)

        geocentric_flat = self.fill_by_bfs(geocentric_flat, current_pose)
        t5 = time.time()
        #print('Time to bfs:', t5 - t4)
        
        return geocentric_flat, agent_view_cropped, agent_view_explored, geocentric_pc

    def update_map(self, depth, current_pose, semantic, goal_category_id):
        depth_for_semantic = depth.copy()
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
            depth_for_semantic[depth_for_semantic > self.vision_range_semantic * self.resolution] = np.NaN

        t0 = time.time()
        self.geocentric_flat, agent_view_cropped, agent_view_explored, _ = self.get_geocentric_flat(depth, self.camera_matrix, \
                                                                                                 self.vision_range, self.resolution, self.map.shape[0], current_pose)
        
        # Inflate obstacles
        self.geocentric_flat[:, :, 1][self.geocentric_flat[:, :, 1] > 0.5] = 1
        self.geocentric_flat[:, :, 1][self.geocentric_flat[:, :, 1] < 0.5] = 0
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
        convolution = convolve2d(self.geocentric_flat[:, :, 1], kernel, mode='same')
        self.geocentric_flat[:, :, 1][convolution == 0] = 0
        i, j = (self.geocentric_flat[:, :, 1] > 0.5).nonzero()
        k = self.obstacle_inflation
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                if di != 0 or dj != 0:
                    ii = np.clip(i + di, 0, self.geocentric_flat.shape[0] - 1)
                    jj = np.clip(j + dj, 0, self.geocentric_flat.shape[1] - 1)
                    self.geocentric_flat[ii, jj, 1] = np.maximum(self.geocentric_flat[ii, jj, 1], 0.5)
        self.geocentric_view = self.geocentric_flat.sum(2)
        t1 = time.time()
        #print('Time to get geocentric flat:', t1 - t0)

        #print(self.map.shape)
        #self.map = self.map + self.geocentric_flat
        self.map = np.maximum(self.map, self.geocentric_flat)
        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt > 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0
        t2 = time.time()
        #print('Time to count map:', t2 - t1)
        #depth[depth is np.NaN] == 6.0 / self.resolution_semantic * 100.

        depth_semantic = np.zeros([self.classes_number, *semantic.shape[:2]])

        nan_map = (semantic == 1).astype(np.float32)
        nan_map[nan_map == 0] = np.NaN
        depth_semantic[0] = depth_for_semantic * nan_map
        #print('MAX SEMANTIC DEPTH:', depth_semantic[depth_semantic == depth_semantic].max())

        self.depth_semantic = depth_semantic.copy()
        t3 = time.time()
        #print('Time to take depth semantic:', t3 - t2)
        
        agent_view_semantic = []
        geocentric_pc_semantic = []
        for index, depth in enumerate(depth_semantic[:1]):
            #print(index, depth.shape)
            semantic_geocentric_flat, agent_view_semantic_cur, _, geocentric_pc = self.get_geocentric_flat(depth, 
                                                                                                           self.semantic_camera_matrix,  
                                                                                                           self.vision_range_semantic, 
                                                                                                           self.resolution_semantic, 
                                                                                                           self.semantic_map.shape[1], 
                                                                                                           current_pose)
            geocentric_pc_semantic.append(geocentric_pc)
            agent_view_semantic.append(agent_view_semantic_cur)
            semantic_geocentric_flat[semantic_geocentric_flat > 0] = 1
            self.semantic_map[index][self.geocentric_view > 0] *= self.goal_decay
            i, j = semantic_geocentric_flat[:, :, 1].nonzero()
            x, y, _ = current_pose
            x_cell = x / self.resolution
            y_cell = y / self.resolution
            dst = np.sqrt((i - y_cell) ** 2 + (j - x_cell) ** 2)
            semantic_geocentric_flat[i[dst < 100 / self.resolution], j[dst < 100 / self.resolution], 1] = 0
            self.semantic_map[index][i[dst < 100 / self.resolution], j[dst < 100 / self.resolution]] /= self.goal_decay
            self.semantic_map[index] = self.semantic_map[index] + semantic_geocentric_flat[:, :, 1]
            
        agent_view_semantic = np.array(agent_view_semantic)
        #print('Goal category id:', goal_category_id)

        self.last_object_depth = depth_semantic[goal_category_id]
        self.last_object_point_cloud = geocentric_pc_semantic[goal_category_id]
        t4 = time.time()
        #print('Time to get agent_view_semantic:', t4 - t3)
        
        #"""
        ################################

        map_semantic_gt = self.semantic_map.copy()
        map_semantic_gt[map_semantic_gt < self.goal_threshold] = 0.0
        map_semantic_gt[map_semantic_gt >= self.goal_threshold] = 1.0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, \
               agent_view_semantic, map_semantic_gt

    def get_info_for_global_reward(self):
        return self.last_object_point_cloud, self.last_object_depth

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]

        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.semantic_map = np.zeros((self.classes_number,
                                      self.map_size_cm // self.resolution_semantic,
                                      self.map_size_cm // self.resolution_semantic,
                                      ), dtype=np.float32)

        self.last_object_point_cloud = None
        self.last_object_depth = None


    def get_map(self):
        return self.map