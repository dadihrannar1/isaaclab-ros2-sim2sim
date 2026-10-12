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
from pynput import keyboard


class WalkForward(Node):
    def __init__(self):
        super().__init__('walk_forward')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.speed = 0.3            # m/s forward


        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        msg = Twist()
        if key == keyboard.Key.right:
            msg.angular.z = -self.speed  # Turn right
        elif key == keyboard.Key.left:
            msg.angular.z = self.speed  # Turn left
        elif key == keyboard.Key.up:
            msg.linear.x = self.speed  # Move forward
        elif key == keyboard.Key.down:
            msg.linear.x = -self.speed  # Move backward
        self.cmd_pub.publish(msg)

    def on_release(self, key):
        # Stop the robot when the key is released
        msg = Twist()
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