#import rospy
import numpy as np
from .frontier_search import FrontierSearch
#from visualization_msgs.msg import Marker, MarkerArray
#from geometry_msgs.msg import Point, Pose
#from std_msgs.msg import ColorRGBA
from copy import deepcopy

class FrontierExploration:
	def __init__(self,
		         mapper,
		         potential_scale=1.0,
		         orientation_scale=1.0,
		         gain_scale=1.0,
		         min_frontier_size=0.25,
		         ):
		self.mapper = mapper
		self.map_resolution = self.mapper.args.map_resolution
		self.potential_scale = potential_scale
		self.orientation_scale = orientation_scale
		self.gain_scale = gain_scale
		self.min_frontier_size = min_frontier_size
		self.frontier_blacklist = []
		self.reached_list = []
		self.current_goal = None
		self.frontier_search = FrontierSearch(self.min_frontier_size, self.map_resolution)
		self.goal_cost = 0
		#self.frontiers_publisher = rospy.Publisher('/explore/frontiers', MarkerArray, latch=True, queue_size=100)


	def reset(self):
		self.frontier_blacklist = []
		self.reached_list = []
		self.current_goal = None


	def reject_goal(self):
		#print('Reject goal:', self.current_goal)
		if self.current_goal is not None:
			self.frontier_blacklist.append(self.current_goal)
		self.current_goal = None


	def accept_goal(self):
		if self.current_goal is not None:
			self.reached_list.append(self.current_goal)
		self.current_goal = None


	def goalOnBlacklist(self, goal):
		goal_x, goal_y = goal
		for x, y in self.frontier_blacklist:
			dst = np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
			if dst < 0.1:
				return True
		return False


	def goalHasReached(self, goal):
		goal_x, goal_y = goal
		for x, y in self.reached_list:
			dst = np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
			if dst < 0.1:
				return True
		return False


	def choose_goal(self, observations):
		x, y = observations['gps']
		y *= -1
		robot_i, robot_j = self.mapper.world_to_map(x, y)
		robot_yaw = observations['compass'][0] * -1
		explored_map = self.mapper.mapper.map[:, :, 1].copy()
		explored_map[self.mapper.mapper.map[:, :, 0] == 0] = -1
		explored_map[self.mapper.mapper.map[:, :, 1] > 0.5] = 1
		semantic_map = self.mapper.semantic_map.copy()
		semantic_map[semantic_map > 0] = -1
		semantic_map[(explored_map > 0) * (semantic_map == 0)] = 1
		#print('Explored map:')
		#print(explored_map[80:95, 130:145])
		#print('Semantic map:')
		#print(semantic_map[80:95, 130:145])
		if semantic_map.min() < 0:
			frontiers = self.frontier_search.searchFrom(robot_i, robot_j, robot_yaw, semantic_map, nav_to_objgoal=True)
		else:
			frontiers = self.frontier_search.searchFrom(robot_i, robot_j, robot_yaw, explored_map, nav_to_objgoal=False)

		for f in frontiers:
			f.cost = self.potential_scale * f.min_distance * self.map_resolution * 0.01 + \
			         self.orientation_scale * f.angle - \
			         self.gain_scale * len(f) * self.map_resolution * 0.01
		frontiers.sort()

		#print('Found {} frontiers'.format(len(frontiers)))
		#for i, f in enumerate(frontiers):
		#	print('Frontier number {} has centroid {} and cost {}'.format(i, f.centroid, f.cost))

		#self.visualize_frontiers(frontiers)
		#print('Blacklist:', self.frontier_blacklist)
		for f in frontiers:
			f_centroid_xy = self.mapper.map_to_world(*f.centroid)
			#print('Centroid:', f_centroid_xy)
			#print('In blacklist and reached list:', self.goalOnBlacklist(f_centroid_xy), self.goalHasReached(f_centroid_xy))
			if not self.goalOnBlacklist(f_centroid_xy) and not self.goalHasReached(f_centroid_xy):
				self.current_goal = f_centroid_xy
				self.goal_cost = f.cost
				print('Choose goal {} with cost {}'.format(self.current_goal, self.goal_cost))
				return self.current_goal
		#print('All frontiers in blacklist. Reset')
		self.reset()
		return None