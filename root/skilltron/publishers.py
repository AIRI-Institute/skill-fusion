import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from cv_bridge import CvBridge
import tf
import numpy as np


class HabitatObservationPublisher:

    def __init__(self,
                 rgb_topic=None,
                 depth_topic=None,
                 true_pose_topic=None,
                 map_topic=None,
                 semantic_map_topic=None,
                 goal_topic=None,
                 path_topic=None,
                 frontier_topic=None):
        self.cvbridge = CvBridge()

        # Initialize RGB image publisher
        if rgb_topic is not None:
            self.publish_rgb = True
            self.image_publisher = rospy.Publisher(rgb_topic, Image, latch=True, queue_size=100)
            self.image = Image()
            self.image.is_bigendian = False
        else:
            self.publish_rgb = False

        # Initialize depth image publisher
        if depth_topic is not None:
            self.publish_depth = True
            self.depth_publisher = rospy.Publisher(depth_topic, Image, latch=True, queue_size=100)
            self.depth = Image()
            self.depth.is_bigendian = True
        else:
            self.publish_depth = False

        # Initialize position publisher
        if true_pose_topic is not None:
            self.publish_true_pose = True
            self.pose_publisher = rospy.Publisher(true_pose_topic, PoseStamped, latch=True, queue_size=100)
            self.tfbr = tf.TransformBroadcaster()
        else:
            self.publish_true_pose = False

        # Initialize map and semantic map publishers
        if map_topic is not None:
            self.publish_map = True
            self.map_publisher = rospy.Publisher(map_topic, OccupancyGrid, latch=True, queue_size=100)
        else:
            self.publish_map = False
        if semantic_map_topic is not None:
            self.publish_semantic_map = True
            self.semantic_map_publisher = rospy.Publisher(semantic_map_topic, OccupancyGrid, latch=True, queue_size=100)
        else:
            self.publish_semantic_map = False

        # Initialize goal publisher
        if goal_topic is not None:
            self.publish_goal = True
            self.goal_publisher = rospy.Publisher(goal_topic, PoseStamped, latch=True, queue_size=100)
        else:
            self.publish_goal = False

        # Initialize path publisher
        if path_topic is not None:
            self.publish_path = True
            self.path_publisher = rospy.Publisher(path_topic, Path, latch=True, queue_size=100)
        else:
            self.publish_path = False


    def publish(self, observations, mapper=None, goal=None, path=None):
        cur_time = rospy.Time.now()

        # Publish RGB image
        if self.publish_rgb:
            self.image = self.cvbridge.cv2_to_imgmsg(observations['rgb'])
            self.image.encoding = 'rgb8'
            self.image.header.stamp = cur_time
            self.image.header.frame_id = 'camera_link'
            self.image_publisher.publish(self.image)

        # Publish depth image
        if self.publish_depth:
            depth = observations['depth'] * 4500 + 500
            depth = depth.astype(np.uint16)
            #print(depth.min(), depth.max())
            self.depth = self.cvbridge.cv2_to_imgmsg(depth)
            self.depth.header.stamp = cur_time
            self.depth.header.frame_id = 'base_scan'
            self.depth_publisher.publish(self.depth)

        # Publish true pose
        if self.publish_true_pose:
            x, y = observations['gps']
            cur_z_angle = observations['compass'][0]
            cur_pose = PoseStamped()
            cur_pose.header.stamp = cur_time
            cur_pose.header.frame_id = 'map'
            cur_pose.pose.position.x = x
            cur_pose.pose.position.y = -y
            cur_pose.pose.position.z = 0
            cur_pose.pose.orientation.x, \
            cur_pose.pose.orientation.y, \
            cur_pose.pose.orientation.z, \
            cur_pose.pose.orientation.w = tf.transformations.quaternion_from_euler(0, 0, cur_z_angle)
            self.tfbr.sendTransform((x, -y, 0),
                                    tf.transformations.quaternion_from_euler(0, 0, cur_z_angle),
                                    cur_time,
                                    'base_link', 'odom')
            self.tfbr.sendTransform((0, 0, 0),
                                    tf.transformations.quaternion_from_euler(0, 0, 0),
                                    cur_time,
                                    'odom', 'map')
            self.pose_publisher.publish(cur_pose)

        # Publish occupancy map
        if self.publish_map and mapper is not None:
            occupancy_map = mapper.mapper.map
            map_msg = OccupancyGrid()
            map_msg.header.stamp = rospy.Time.now()
            map_msg.header.frame_id = 'map'
            map_msg.info.resolution = mapper.mapper.resolution / 100.
            map_msg.info.width = occupancy_map.shape[1]
            map_msg.info.height = occupancy_map.shape[0]
            map_msg.info.origin.position.x = -occupancy_map.shape[1] * mapper.mapper.resolution / 200.
            map_msg.info.origin.position.y = -occupancy_map.shape[0] * mapper.mapper.resolution / 200.
            map_data = np.ones((map_msg.info.height, map_msg.info.width), dtype=np.int8) * (-1)
            map_data[occupancy_map[:, :, 0] > 0] = 0
            map_data[occupancy_map[:, :, 1] > 0] = 100
            map_msg.data = list(map_data.ravel())
            self.map_publisher.publish(map_msg)

        # Publish semantic map
        if self.publish_semantic_map and mapper is not None:
            occupancy_map = mapper.mapper.map
            semantic_map = mapper.semantic_map
            map_msg.header.stamp = rospy.Time.now()
            map_msg.header.frame_id = 'map'
            map_msg.info.resolution = mapper.mapper.resolution / 100.
            map_msg.info.width = semantic_map.shape[1]
            map_msg.info.height = semantic_map.shape[0]
            map_msg.info.origin.position.x = -semantic_map.shape[1] * mapper.mapper.resolution / 200.
            map_msg.info.origin.position.y = -semantic_map.shape[0] * mapper.mapper.resolution / 200.
            map_data = np.zeros((map_msg.info.height, map_msg.info.width), dtype=np.int8)
            map_data[occupancy_map[:, :, 1] > 0] = 100
            map_data[semantic_map > 0] = -1
            map_msg.data = list(map_data.ravel())
            self.semantic_map_publisher.publish(map_msg)

        # Publish exploration goal
        if self.publish_goal and goal is not None:
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = 'odom'
            goal_msg.header.stamp = rospy.Time.now()
            x, y = goal
            goal_msg.pose.position.x = x
            goal_msg.pose.position.y = y
            self.goal_publisher.publish(goal_msg)

        # Publish path to goal
        if self.publish_path and path is not None:
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = 'odom'
            path_msg.poses = []
            for x, y in path:
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = x
                pose_msg.pose.position.y = y
                path_msg.poses.append(pose_msg)
            self.path_publisher.publish(path_msg)