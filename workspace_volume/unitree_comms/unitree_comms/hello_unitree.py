import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowState

class HelloG1(Node):
    def __init__(self):
        super().__init__('hello_g1')
        self.create_subscription(LowState, 'lowstate', self.cb, 10)
        self.count = 0

    def cb(self, msg: LowState):
        self.count += 1
        if self.count % 50:          # lowstate is ~500 Hz, print ~10x/sec
            return
        r, p, y = msg.imu_state.rpy
        self.get_logger().info(
            f'Hello G1! roll={r:+.2f} pitch={p:+.2f} yaw={y:+.2f} '
            f'joint0={msg.motor_state[0].q:+.3f} rad')

def main():
    rclpy.init()
    rclpy.spin(HelloG1())

if __name__ == '__main__':
    main()