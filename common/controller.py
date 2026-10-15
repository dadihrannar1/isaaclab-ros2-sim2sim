#!/usr/bin/env python3
"""Keyboard teleop for the sim2sim robots: publishes /cmd_vel from the keys currently held down.

Keys can be combined (e.g. up + left walks forward while turning).

    up / down      forward / backward
    left / right   turn left / right
    a / d          strafe left / right
    + / -          speed up / down (= also works as +, no shift needed)
    space          release all keys (stop)
    q              quit
"""
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pynput import keyboard

# Max command per axis at 100% speed. Keep within the ranges the policy was trained on.
MAX_VX = 1.0    # m/s forward/backward
MAX_VY = 0.5    # m/s sideways
MAX_YAW = 1.0   # rad/s turning

SPEED_STEP = 0.1
SPEED_MIN = 0.1
SPEED_MAX = 1.0

PUBLISH_HZ = 20.0


class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.speed = 0.3            # fraction of the max command, adjusted with +/-
        self.held = set()           # keys currently held down
        self.lock = threading.Lock()  # pynput calls back from its own thread
        self.quit = False
        self.last_status = None

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # Publish continuously so the command reflects the held keys, independent of key-repeat
        self.create_timer(1.0 / PUBLISH_HZ, self.publish_cmd)

        print(__doc__)
        self.print_status(Twist())

    @staticmethod
    def normalize(key):
        """Turn a pynput key into a comparable value: Key.* for special keys, lowercase char otherwise."""
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        return key

    def on_press(self, key):
        k = self.normalize(key)
        with self.lock:
            if k in ('+', '='):
                self.speed = min(SPEED_MAX, round(self.speed + SPEED_STEP, 2))
            elif k == '-':
                self.speed = max(SPEED_MIN, round(self.speed - SPEED_STEP, 2))
            elif k == keyboard.Key.space:
                self.held.clear()
            # make q quit
            elif k == 'q':
                self.quit = True
            else:
                self.held.add(k)

    def on_release(self, key):
        with self.lock:
            self.held.discard(self.normalize(key))

    def axis(self, positive, negative):
        """+1, -1 or 0 depending on which of the two keys is held (both cancel out)."""
        return (positive in self.held) - (negative in self.held)

    def publish_cmd(self):
        if self.quit:
            raise SystemExit

        msg = Twist()
        with self.lock:
            msg.linear.x = self.speed * MAX_VX * self.axis(keyboard.Key.up, keyboard.Key.down)
            msg.linear.y = self.speed * MAX_VY * self.axis('a', 'd')
            msg.angular.z = self.speed * MAX_YAW * self.axis(keyboard.Key.left, keyboard.Key.right)
        self.cmd_pub.publish(msg)
        self.print_status(msg)

    def print_status(self, msg):
        status = (f'speed {self.speed * 100:3.0f}%  |  '
                  f'vx {msg.linear.x:+.2f}  vy {msg.linear.y:+.2f}  yaw {msg.angular.z:+.2f}')
        if status != self.last_status:
            print('\r' + status + '   ', end='', flush=True)
            self.last_status = status


def main():
    rclpy.init()
    node = Controller()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        # Make sure we stop the robot
        node.listener.stop()
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        print('\nDone.')


if __name__ == '__main__':
    main()
