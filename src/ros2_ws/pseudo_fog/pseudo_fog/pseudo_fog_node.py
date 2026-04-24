#!/usr/bin/env python3
import math
import random
from typing import Optional, List

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class PseudoFogNode(Node):
    def __init__(self) -> None:
        super().__init__('pseudo_fog_node')

        # =========================
        # 最簡版參數
        # =========================
        self.declare_parameter('input_odom_topic', '/odom')
        self.declare_parameter('output_imu_topic', '/pseudo_fog/imu')
        self.declare_parameter('output_rate_hz', 82.0)

        # 高等級 FOG 第一版近似
        self.declare_parameter('tau', 2000.0)                  # sec
        self.declare_parameter('bias_stationary_std', 5e-08)   # rad/s
        self.declare_parameter('white_noise_std', 1.58e-05)    # rad/s per sample

        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('use_fixed_seed', True)
        self.declare_parameter('random_seed', 42)

        self.input_odom_topic = self.get_parameter('input_odom_topic').value
        self.output_imu_topic = self.get_parameter('output_imu_topic').value
        self.output_rate_hz = float(self.get_parameter('output_rate_hz').value)

        self.tau = float(self.get_parameter('tau').value)
        self.bias_stationary_std = float(self.get_parameter('bias_stationary_std').value)
        self.white_noise_std = float(self.get_parameter('white_noise_std').value)

        self.frame_id = self.get_parameter('frame_id').value
        self.use_fixed_seed = bool(self.get_parameter('use_fixed_seed').value)
        self.random_seed = int(self.get_parameter('random_seed').value)

        if self.output_rate_hz <= 0.0:
            raise ValueError('output_rate_hz 必須 > 0')
        if self.tau <= 0.0:
            raise ValueError('tau 必須 > 0')
        if self.bias_stationary_std < 0.0:
            raise ValueError('bias_stationary_std 必須 >= 0')
        if self.white_noise_std < 0.0:
            raise ValueError('white_noise_std 必須 >= 0')

        if self.use_fixed_seed:
            random.seed(self.random_seed)

        self.dt = 1.0 / self.output_rate_hz

        # 最新 GT angular velocity
        self.latest_odom: Optional[Odometry] = None
        self.latest_gt_omega: List[float] = [0.0, 0.0, 0.0]

        # GM1 bias state
        self.bias: List[float] = [0.0, 0.0, 0.0]

        # =========================
        # ROS interfaces
        # =========================
        self.odom_sub = self.create_subscription(
            Odometry,
            self.input_odom_topic,
            self.odom_callback,
            50
        )

        self.imu_pub = self.create_publisher(
            Imu,
            self.output_imu_topic,
            50
        )

        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info('Pseudo FOG node started.')
        self.get_logger().info(f'  input odom topic  : {self.input_odom_topic}')
        self.get_logger().info(f'  output imu topic  : {self.output_imu_topic}')
        self.get_logger().info(f'  output rate (Hz)  : {self.output_rate_hz}')
        self.get_logger().info(f'  tau (s)           : {self.tau}')
        self.get_logger().info(f'  bias std (rad/s)  : {self.bias_stationary_std}')
        self.get_logger().info(f'  white std (rad/s) : {self.white_noise_std}')
        self.get_logger().info(f'  frame_id          : {self.frame_id}')

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.latest_gt_omega[0] = msg.twist.twist.angular.x
        self.latest_gt_omega[1] = msg.twist.twist.angular.y
        self.latest_gt_omega[2] = msg.twist.twist.angular.z

    def update_gm1_bias(self) -> None:
        """
        b_{k+1} = phi * b_k + eta_k
        phi = exp(-dt / tau)
        eta_k ~ N(0, sigma_eta^2)
        sigma_eta = sigma_b * sqrt(1 - phi^2)
        """
        phi = math.exp(-self.dt / self.tau)
        sigma_eta = self.bias_stationary_std * math.sqrt(max(0.0, 1.0 - phi * phi))

        for i in range(3):
            eta = random.gauss(0.0, sigma_eta)
            self.bias[i] = phi * self.bias[i] + eta

    def timer_callback(self) -> None:
        if self.latest_odom is None:
            return

        # 更新 GM1 bias
        self.update_gm1_bias()

        wx_gt, wy_gt, wz_gt = self.latest_gt_omega

        # 最簡版：GT + GM1 bias + white noise
        wx = wx_gt + self.bias[0] + random.gauss(0.0, self.white_noise_std)
        wy = wy_gt + self.bias[1] + random.gauss(0.0, self.white_noise_std)
        wz = wz_gt + self.bias[2] + random.gauss(0.0, self.white_noise_std)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # 不提供 orientation
        msg.orientation_covariance = [
            -1.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0
        ]

        msg.angular_velocity.x = wx
        msg.angular_velocity.y = wy
        msg.angular_velocity.z = wz

        ang_var = self.white_noise_std ** 2 + self.bias_stationary_std ** 2
        msg.angular_velocity_covariance = [
            ang_var, 0.0,    0.0,
            0.0,    ang_var, 0.0,
            0.0,    0.0,    ang_var
        ]

        # 不提供 linear acceleration
        msg.linear_acceleration_covariance = [
            -1.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0
        ]

        self.imu_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PseudoFogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()