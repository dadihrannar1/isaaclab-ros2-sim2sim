#ROS node subscribes to goal_pose and odometry, and publishes cmd_vel until reaching the goal.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import math

class WalkToGoal(Node):
    def __init__(self):
        super().__init__('walk_to_goal')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.start_x = None
        self.start_y = None
        self.start_orien_z = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orien_z = 0.0
        self.speed = 0.3            # m/s forward
        self.done = False
        self.goal_x = None
        self.goal_y = None
        self.goal_orien_z = None

        #PID gains
        self.kp = 1.0
        self.ki = 0.01
        self.kd = 0.0
        self.kp_theta = 1.0
        self.integral_limit = 5.0   # anti-windup: cap on the summed error

        # command limits: slow, calm walking (training ranges in flat_env_cfg.py are vx ±1.0, vy ±0.5, wz ±1.0)
        self.max_vx = 0.4
        self.max_vy = 0.3
        self.max_wz = 0.5

        self.prev_x_error = 0.0
        self.prev_y_error = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.derivative_x = 0.0
        self.derivative_y = 0.0

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Waiting for odometry...')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_orien_z = yaw_from_quat(msg.pose.pose.orientation)

        if self.start_x is None:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.get_logger().info(
                f'Start position: ({self.start_x:.3f}, {self.start_y:.3f})')

    def goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_orien_z = yaw_from_quat(msg.pose.orientation)

        # new goal: start the controller fresh
        self.done = False
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_x_error = 0.0
        self.prev_y_error = 0.0
        self.get_logger().info(
            f'Goal position: ({self.goal_x:.3f}, {self.goal_y:.3f})')

    def control_loop(self):
        msg = Twist()

        if self.done:
            # keep sending zero so the robot stands still
            self.cmd_pub.publish(msg)
            return

        if self.start_x is None or self.goal_x is None:
            # Still waiting for first odom message
            self.cmd_pub.publish(msg)
            return

        distance = math.sqrt(
            (self.current_x - self.goal_x) ** 2 +
            (self.current_y - self.goal_y) ** 2
        )

        #pos and orientation feedback in robot's frame
        self.x_error = (self.goal_x - self.current_x) * math.cos(self.current_orien_z) + (self.goal_y - self.current_y) * math.sin(self.current_orien_z)
        self.y_error = (self.goal_y - self.current_y) * math.cos(self.current_orien_z) - (self.goal_x - self.current_x) * math.sin(self.current_orien_z)
        self.theta_error = wrap_angle(self.goal_orien_z - self.current_orien_z)

        #integral and derivative terms (integral clamped against windup)
        self.integral_x = clamp(self.integral_x + self.x_error, self.integral_limit)
        self.integral_y = clamp(self.integral_y + self.y_error, self.integral_limit)

        self.derivative_x = self.x_error - self.prev_x_error
        self.derivative_y = self.y_error - self.prev_y_error
        self.prev_x_error = self.x_error
        self.prev_y_error = self.y_error

        self.get_logger().info(
            f'Distance to goal: {distance:.3f} m')

        if distance < 0.1 and abs(self.theta_error) < 0.1:
            self.get_logger().info('Reached the goal! Waiting for a new goal on /goal_pose...')
            self.done = True
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.0
        else:
            # pid control for x, y, and theta errors
            vx = self.kp * self.x_error + self.ki * self.integral_x + self.kd * self.derivative_x
            vy = self.kp * self.y_error + self.ki * self.integral_y + self.kd * self.derivative_y
            wz = self.kp_theta * self.theta_error  # simple proportional control for orientation

            # keep commands inside the policy's training range
            msg.linear.x = clamp(vx, self.max_vx)
            msg.linear.y = clamp(vy, self.max_vy)
            msg.angular.z = clamp(wz, self.max_wz)

        self.cmd_pub.publish(msg)


def yaw_from_quat(q):
    """Yaw (rad) from a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(a):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def clamp(value, limit):
    return max(-limit, min(limit, value))



def main(args=None):
    rclpy.init(args=args)
    walk_to_goal_node = WalkToGoal()
    rclpy.spin(walk_to_goal_node)
    walk_to_goal_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()