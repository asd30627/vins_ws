#!/usr/bin/env python3
import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class FogCompareNode(Node):
    def __init__(self) -> None:
        super().__init__('fog_compare_node')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('fog_topic', '/pseudo_fog/imu')
        self.declare_parameter('duration_sec', 10.0)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.fog_topic = self.get_parameter('fog_topic').value
        self.duration_sec = float(self.get_parameter('duration_sec').value)

        self.latest_odom_omega: Optional[Tuple[float, float, float]] = None
        self.samples: List[Tuple[float, float, float]] = []

        self.start_time = self.get_clock().now()

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            50
        )

        self.fog_sub = self.create_subscription(
            Imu,
            self.fog_topic,
            self.fog_callback,
            50
        )

        self.timer = self.create_timer(0.2, self.check_done)

        self.get_logger().info('Fog compare node started')
        self.get_logger().info(f'odom topic    : {self.odom_topic}')
        self.get_logger().info(f'fog topic     : {self.fog_topic}')
        self.get_logger().info(f'duration (s)  : {self.duration_sec}')

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_odom_omega = (
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        )

    def fog_callback(self, msg: Imu) -> None:
        if self.latest_odom_omega is None:
            return

        ox, oy, oz = self.latest_odom_omega
        fx = msg.angular_velocity.x
        fy = msg.angular_velocity.y
        fz = msg.angular_velocity.z

        ex = fx - ox
        ey = fy - oy
        ez = fz - oz

        self.samples.append((ex, ey, ez))

    def check_done(self) -> None:
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < self.duration_sec:
            return

        self.print_report()
        self.destroy_node()
        rclpy.shutdown()

    def print_report(self) -> None:
        n = len(self.samples)
        if n == 0:
            self.get_logger().warn('沒有收集到任何可比較樣本，請確認 /odom 與 /pseudo_fog/imu 都有在發布。')
            return

        exs = [s[0] for s in self.samples]
        eys = [s[1] for s in self.samples]
        ezs = [s[2] for s in self.samples]

        self.report_axis('x', exs)
        self.report_axis('y', eys)
        self.report_axis('z', ezs)

        mag = [math.sqrt(x*x + y*y + z*z) for x, y, z in self.samples]
        mean_mag = sum(mag) / n
        rmse_mag = math.sqrt(sum(v*v for v in mag) / n)
        max_mag = max(mag)

        self.get_logger().info('================ Overall =================')
        self.get_logger().info(f'samples           : {n}')
        self.get_logger().info(f'mean |e| norm     : {mean_mag:.8f} rad/s')
        self.get_logger().info(f'RMSE error norm   : {rmse_mag:.8f} rad/s')
        self.get_logger().info(f'max error norm    : {max_mag:.8f} rad/s')

    def report_axis(self, axis_name: str, values: List[float]) -> None:
        n = len(values)
        mean_err = sum(values) / n
        mean_abs = sum(abs(v) for v in values) / n
        rmse = math.sqrt(sum(v * v for v in values) / n)
        max_abs = max(abs(v) for v in values)

        self.get_logger().info(f'================ Axis {axis_name} =================')
        self.get_logger().info(f'bias mean         : {mean_err:.8f} rad/s')
        self.get_logger().info(f'mean abs error    : {mean_abs:.8f} rad/s')
        self.get_logger().info(f'RMSE              : {rmse:.8f} rad/s')
        self.get_logger().info(f'max abs error     : {max_abs:.8f} rad/s')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FogCompareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()