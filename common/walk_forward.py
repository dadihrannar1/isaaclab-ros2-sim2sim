#!/usr/bin/env python3
"""
Walk the simulated robot forward 1 meter then stop. Robot-agnostic: drives any
sim publishing /cmd_vel-driven locomotion and /odom (both the A1 and G1 deployment
scripts in this repo use the same topic names).
Run in Terminal 2 (ROS 2 sourced, NOT the Isaac Sim terminal).

Usage:
    source /opt/ros/humble/setup.bash
    python3 common/walk_forward.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math


class WalkForward(Node):
    def __init__(self):
        super().__init__('walk_forward')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.start_x = None
        self.start_y = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.target_distance = 1.0  # meters
        self.speed = 0.3            # m/s forward
        self.done = False

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Waiting for odometry...')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(
                f'Start position: ({self.start_x:.3f}, {self.start_y:.3f})')

    def control_loop(self):
        if self.done:
            return

        msg = Twist()

        if self.start_x is None:
            # Still waiting for first odom message
            self.cmd_pub.publish(msg)
            return

        distance = math.sqrt(
            (self.current_x - self.start_x) ** 2 +
            (self.current_y - self.start_y) ** 2
        )

        self.get_logger().info(
            f'Distance: {distance:.3f} / {self.target_distance} m',
            throttle_duration_sec=0.5)

        if distance >= self.target_distance:
            self.cmd_pub.publish(msg)  # publish zero velocity to stop
            self.get_logger().info(
                f'Target reached at ({self.current_x:.3f}, {self.current_y:.3f})! Stopping.')
            self.done = True
            raise SystemExit
        else:
            msg.linear.x = self.speed
            self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = WalkForward()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        # Make sure we stop the robot
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()
        print('Done.')


if __name__ == '__main__':
    main()