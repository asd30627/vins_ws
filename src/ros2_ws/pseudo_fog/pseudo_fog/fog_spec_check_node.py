#!/usr/bin/env python3
import math
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class FogSpecCheckNode(Node):
    def __init__(self) -> None:
        super().__init__('fog_spec_check_node')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('fog_topic', '/pseudo_fog/imu')

        # 建議至少 1800s(30分鐘)，更好是 3600s(1小時) 以上
        self.declare_parameter('duration_sec', 1800.0)

        # 是否使用 GT 靜止篩選
        self.declare_parameter('use_gt_stationary_gate', True)

        # 放寬預設門檻
        self.declare_parameter('gt_stationary_threshold_rad_s', 1e-3)

        # 規格門檻（來自你的簡報）
        self.declare_parameter('target_arw_deg_per_sqrt_hr', 0.006)
        self.declare_parameter('target_bias_instability_deg_per_hr', 0.01)

        # Allan deviation 掃描點數
        self.declare_parameter('num_cluster_points', 30)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.fog_topic = self.get_parameter('fog_topic').value
        self.duration_sec = float(self.get_parameter('duration_sec').value)

        self.use_gt_stationary_gate = bool(self.get_parameter('use_gt_stationary_gate').value)
        self.gt_stationary_threshold_rad_s = float(self.get_parameter('gt_stationary_threshold_rad_s').value)

        self.target_arw_deg_per_sqrt_hr = float(self.get_parameter('target_arw_deg_per_sqrt_hr').value)
        self.target_bias_instability_deg_per_hr = float(self.get_parameter('target_bias_instability_deg_per_hr').value)
        self.num_cluster_points = int(self.get_parameter('num_cluster_points').value)

        self.latest_odom_omega: Optional[Tuple[float, float, float]] = None

        self.timestamps_sec: List[float] = []
        self.err_x: List[float] = []
        self.err_y: List[float] = []
        self.err_z: List[float] = []

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

        self.timer = self.create_timer(0.5, self.check_done)

        self.get_logger().info('Fog spec check node started')
        self.get_logger().info(f'odom topic                     : {self.odom_topic}')
        self.get_logger().info(f'fog topic                      : {self.fog_topic}')
        self.get_logger().info(f'duration (s)                   : {self.duration_sec}')
        self.get_logger().info(f'use GT stationary gate         : {self.use_gt_stationary_gate}')
        self.get_logger().info(f'GT stationary threshold rad/s  : {self.gt_stationary_threshold_rad_s}')
        self.get_logger().info(f'target ARW deg/sqrt(hr)        : {self.target_arw_deg_per_sqrt_hr}')
        self.get_logger().info(f'target bias instability deg/hr : {self.target_bias_instability_deg_per_hr}')

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

        if self.use_gt_stationary_gate:
            gt_norm = math.sqrt(ox * ox + oy * oy + oz * oz)
            if gt_norm > self.gt_stationary_threshold_rad_s:
                return

        fx = msg.angular_velocity.x
        fy = msg.angular_velocity.y
        fz = msg.angular_velocity.z

        ex = fx - ox
        ey = fy - oy
        ez = fz - oz

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.timestamps_sec.append(t)
        self.err_x.append(ex)
        self.err_y.append(ey)
        self.err_z.append(ez)

    def check_done(self) -> None:
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < self.duration_sec:
            return

        self.print_report_and_shutdown()

    def print_report_and_shutdown(self) -> None:
        n = len(self.timestamps_sec)
        if n < 100:
            self.get_logger().warn('有效樣本太少，無法做規格判定。請確認載具靜止、/odom 與 /pseudo_fog/imu 都有發布。')
            self.destroy_node()
            rclpy.shutdown()
            return

        ts = np.array(self.timestamps_sec, dtype=np.float64)
        ex = np.array(self.err_x, dtype=np.float64)
        ey = np.array(self.err_y, dtype=np.float64)
        ez = np.array(self.err_z, dtype=np.float64)

        dts = np.diff(ts)
        dts = dts[dts > 0]
        if len(dts) == 0:
            self.get_logger().warn('時間戳無法估計 dt。')
            self.destroy_node()
            rclpy.shutdown()
            return

        dt = float(np.median(dts))
        fs = 1.0 / dt

        self.get_logger().info('================ Test Summary =================')
        self.get_logger().info(f'valid samples             : {n}')
        self.get_logger().info(f'estimated dt              : {dt:.8f} s')
        self.get_logger().info(f'estimated fs              : {fs:.3f} Hz')

        self.print_basic_stats('x', ex)
        self.print_basic_stats('y', ey)
        self.print_basic_stats('z', ez)

        self.check_axis_spec('x', ex, dt)
        self.check_axis_spec('y', ey, dt)
        self.check_axis_spec('z', ez, dt)

        self.destroy_node()
        rclpy.shutdown()

    def print_basic_stats(self, axis: str, e: np.ndarray) -> None:
        mean_err = float(np.mean(e))
        std_err = float(np.std(e))
        max_abs = float(np.max(np.abs(e)))

        self.get_logger().info(f'---------------- Axis {axis} basic ----------------')
        self.get_logger().info(f'mean residual       : {mean_err:.10e} rad/s')
        self.get_logger().info(f'std residual        : {std_err:.10e} rad/s')
        self.get_logger().info(f'max abs residual    : {max_abs:.10e} rad/s')

    def check_axis_spec(self, axis: str, e: np.ndarray, dt: float) -> None:
        tau, adev = self.compute_allan_deviation_rate(e, dt, self.num_cluster_points)

        if len(tau) < 3:
            self.get_logger().warn(f'Axis {axis}: Allan deviation 點數不足，無法判規。')
            return

        arw_deg_per_sqrt_hr = self.estimate_arw_from_adev(tau, adev)
        bias_deg_per_hr = self.estimate_bias_instability_from_adev(tau, adev)

        arw_pass = arw_deg_per_sqrt_hr <= self.target_arw_deg_per_sqrt_hr
        bias_pass = bias_deg_per_hr <= self.target_bias_instability_deg_per_hr

        self.get_logger().info(f'================ Axis {axis} spec check =================')
        self.get_logger().info(f'ARW estimate               : {arw_deg_per_sqrt_hr:.8f} deg/sqrt(hr)')
        self.get_logger().info(f'ARW target                 : {self.target_arw_deg_per_sqrt_hr:.8f} deg/sqrt(hr)')
        self.get_logger().info(f'ARW PASS                   : {arw_pass}')
        self.get_logger().info(f'Bias instability estimate  : {bias_deg_per_hr:.8f} deg/hr')
        self.get_logger().info(f'Bias instability target    : {self.target_bias_instability_deg_per_hr:.8f} deg/hr')
        self.get_logger().info(f'Bias PASS                  : {bias_pass}')

    def compute_allan_deviation_rate(self, omega: np.ndarray, dt: float, num_points: int):
        L = len(omega)
        if L < 10:
            return np.array([]), np.array([])

        theta = np.cumsum(omega) * dt

        m_max = (L - 1) // 2
        if m_max < 1:
            return np.array([]), np.array([])

        m_values = np.unique(
            np.clip(
                np.round(np.logspace(0, math.log10(m_max), num=num_points)).astype(int),
                1,
                m_max
            )
        )

        tau_list = []
        adev_list = []

        for m in m_values:
            if L - 2 * m <= 0:
                continue

            diff2 = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
            avar = np.mean(diff2 ** 2) / (2.0 * (m * dt) ** 2)
            adev = math.sqrt(avar)

            tau_list.append(m * dt)
            adev_list.append(adev)

        return np.array(tau_list), np.array(adev_list)

    def estimate_arw_from_adev(self, tau: np.ndarray, adev: np.ndarray) -> float:
        logtau = np.log10(tau)
        logadev = np.log10(adev)
        slopes = np.diff(logadev) / np.diff(logtau)

        i = int(np.argmin(np.abs(slopes - (-0.5))))
        b = logadev[i] - (-0.5) * logtau[i]
        N_rad_per_sqrt_s = 10 ** b

        # rad/sqrt(s) -> deg/sqrt(hr)
        N_deg_per_sqrt_hr = N_rad_per_sqrt_s * 180.0 / math.pi * 60.0
        return float(N_deg_per_sqrt_hr)

    def estimate_bias_instability_from_adev(self, tau: np.ndarray, adev: np.ndarray) -> float:
        logtau = np.log10(tau)
        logadev = np.log10(adev)
        slopes = np.diff(logadev) / np.diff(logtau)

        i = int(np.argmin(np.abs(slopes - 0.0)))
        sigma_flat = adev[i]

        scfB = math.sqrt(2.0 * math.log(2.0) / math.pi)  # ~0.664
        B_rad_per_s = sigma_flat / scfB

        # rad/s -> deg/hr
        B_deg_per_hr = B_rad_per_s * 180.0 / math.pi * 3600.0
        return float(B_deg_per_hr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FogSpecCheckNode()
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