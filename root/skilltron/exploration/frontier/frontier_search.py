import numpy as np
from heapq import heappush, heappop
from collections import deque
from time import time


def normalize(angle):
	while angle > np.pi:
		angle -= 2 * np.pi
	while angle < -np.pi:
		angle += 2 * np.pi
	return angle


class Frontier:
	def __init__(self, points, min_distance, angle):
		self.points = points
		self.centroid = np.mean(points, axis=0)
		self.min_distance = min_distance
		self.angle = angle


	def __lt__(self, other):
		return self.cost < other.cost


	def __eq__(self, other):
		return


	def __len__(self):
		return len(self.points)


class FrontierSearch:
	def __init__(self, min_frontier_size, map_resolution):
		self.min_frontier_size = min_frontier_size
		self.map_resolution = map_resolution
		self.grid_map = None
		self.frontier_flag = None
		self.n_checked_nodes = 0


	def isNewFrontierCell(self, i, j):
		if i < 0 or i >= self.grid_map.shape[0]:
			return False
		if j < 0 or j >= self.grid_map.shape[1]:
			return False
		if self.grid_map[i, j] >= 0:
			#if i > 88 and i < 93 and j > 130 and j < 145:
			#	print('({}, {}) is not new frontier cell because of map'.format(i, j))
			return False
		if self.frontier_flag[i, j]:
			#if i > 88 and i < 93 and j > 130 and j < 145:
			#	print('({}, {}) is not new frontier cell because of flag'.format(i, j))
			return False
		self.n_checked_nodes += 1
		for di in range(-1, 2):
			for dj in range(-1, 2):
				if di == 0 and dj == 0:
					continue
				if i + di < 0 or i + di >= self.grid_map.shape[0]:
					continue
				if j + dj < 0 or j + dj >= self.grid_map.shape[1]:
					continue
				if self.grid_map[i + di, j + dj] == 0:
					#if i > 88 and i < 93 and j > 130 and j < 145:
					#	print('Is new frontier cell ({}, {})'.format(i, j))
					#	print(self.grid_map[80:95, 130:145])
					return True
		#if i > 88 and i < 93 and j > 130 and j < 145:
		#	print('({}, {}) is not new frontier cell because of no neighbors'.format(i, j))
		return False


	def buildNewFrontier(self, init_i, init_j, robot_i, robot_j, robot_yaw):
		#print('Build new frontier around ({}, {})'.format(init_i, init_j))
		points = []
		min_distance = np.inf
		angle = 0
		bfs = deque()
		bfs.append((init_i, init_j))
		while len(bfs) > 0:
			i, j = bfs.popleft()
			for di in range(-1, 2):
				for dj in range(-1, 2):
					if di == 0 and dj == 0:
						continue
					if self.isNewFrontierCell(i + di, j + dj):
						bfs.append((i + di, j + dj))
						self.frontier_flag[i + di, j + dj] = True
						points.append((i + di, j + dj))
						dst = np.sqrt((i + di - robot_i) ** 2 + (j + dj - robot_j) ** 2)
						if dst < min_distance:
							min_distance = dst
							direction_to_frontier = np.arctan2(i + di - robot_i, j + dj - robot_j)
							angle = normalize(direction_to_frontier - robot_yaw)
		return Frontier(points, min_distance, angle)



	def searchFrom(self, robot_i, robot_j, robot_yaw, grid_map, nav_to_objgoal=False):
		#print('Search from ({}, {})'.format(robot_i, robot_j))
		self.grid_map = grid_map
		self.frontier_flag = np.zeros_like(self.grid_map)
		distance = np.ones_like(self.grid_map) * np.inf
		distance[robot_i, robot_j] = 0
		prev = np.ones((self.grid_map.shape[0], self.grid_map.shape[1], 2)) * -1

		heap = []
		frontier_list = []
		heappush(heap, (0, (robot_i, robot_j)))
		cnt = 0
		self.n_checked_nodes = 0
		frontier_found = False
		while len(heap) > 0:
			dst, pos = heappop(heap)
			i, j = pos
			cnt += 1
			if frontier_found and nav_to_objgoal and distance[i, j] > 50:
				continue
			for di in range(-1, 2):
				for dj in range(-1, 2):
					if di == 0 and dj == 0:
						continue
					if i + di < 0 or i + di >= distance.shape[0]:
						continue
					if j + dj < 0 or j + dj >= distance.shape[1]:
						continue
					step = np.sqrt(di ** 2 + dj ** 2)
					if self.grid_map[i + di, j + dj] == 0 and distance[i + di, j + dj] > distance[i, j] + step:
						distance[i + di, j + dj] = distance[i, j] + step
						prev[i + di, j + dj] = [i, j]
						heappush(heap, (distance[i + di, j + dj], (i + di, j + dj)))
					elif self.isNewFrontierCell(i + di, j + dj):
						self.frontier_flag[i + di, j + dj] = True
						new_frontier = self.buildNewFrontier(i + di, j + dj, robot_i, robot_j, robot_yaw)
						frontier_found = True
						new_frontier.min_distance = distance[i, j] + step
						if len(new_frontier) * self.map_resolution / 100. >= self.min_frontier_size:
							frontier_list.append(new_frontier)
		#print('Explored {} nodes'.format(cnt))
		#print('Checked for frontier {} nodes'.format(cnt))
		return frontier_list