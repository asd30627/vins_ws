import csv
import os
import re
import time
import statistics
import math
from bisect import bisect_left
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, Imu
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


@dataclass
class Event:
    ts_ns: int
    kind: str  # 'imu' or 'stereo' or 'gt'
    payload: object


@dataclass
class StereoFrame:
    ts_ns: int
    left_path: str
    right_path: str


@dataclass
class ImuSample:
    ts_ns: int
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass
class FogGyroSample:
    ts_ns: int
    gx: float
    gy: float
    gz: float


@dataclass
class PoseSample:
    ts_ns: int
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def normalize_header(s: str) -> str:
    return s.strip().lower().replace(' ', '_').replace('-', '_')


def sec_nsec_from_ns(ts_ns: int) -> Tuple[int, int]:
    sec = ts_ns // 1_000_000_000
    nsec = ts_ns % 1_000_000_000
    return sec, nsec


def parse_int_lossless(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None

    if re.fullmatch(r'[+-]?\d+', s):
        try:
            return int(s)
        except Exception:
            return None

    try:
        return int(Decimal(s))
    except (InvalidOperation, ValueError):
        return None


def quaternion_from_rotation_matrix(
    r00: float, r01: float, r02: float,
    r10: float, r11: float, r12: float,
    r20: float, r21: float, r22: float
) -> Tuple[float, float, float, float]:
    trace = r00 + r11 + r22

    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = (1.0 + r00 - r11 - r22) ** 0.5 * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = (1.0 + r11 - r00 - r22) ** 0.5 * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = (1.0 + r22 - r00 - r11) ** 0.5 * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0

    return qx / norm, qy / norm, qz / norm, qw / norm


class KaistPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__('kaist_player_node')

        self.declare_parameter(
            'dataset_root',
            '/mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo'
        )
        self.declare_parameter(
            'stereo_stamp_csv',
            'data/urban28-pankyo/sensor_data/stereo_stamp.csv'
        )
        self.declare_parameter(
            'imu_csv',
            'data/urban28-pankyo/sensor_data/xsens_imu.csv'
        )
        self.declare_parameter(
            'fog_csv',
            'data/urban28-pankyo/sensor_data/fog.csv'
        )
        self.declare_parameter('image_root', 'img')

        # imu source mode
        # xsens      : accel + gyro both from xsens_imu.csv
        # fog_xsens  : accel from xsens_imu.csv, gyro from fog.csv (nearest timestamp)
        self.declare_parameter('imu_source', 'xsens')
        # fog gyro conversion options
        # KAIST fog.csv 很可能是每個 sample 的角度增量，而不是角速度
        # 所以預設會用相鄰 timestamp 算 dt，做 raw/dt -> rad/s
        self.declare_parameter('fog_use_dt_rate', True)
        self.declare_parameter('fog_scale', 1.0)
        self.declare_parameter('fog_axis_permutation', '0,1,2')
        self.declare_parameter('fog_axis_signs', '1,1,1')
        self.declare_parameter('fog_auto_bias', True)
        self.declare_parameter('fog_bias_estimation_duration_sec', 5.0)

        self.declare_parameter('left_keywords', 'left,stereo_left,cam0')
        self.declare_parameter('right_keywords', 'right,stereo_right,cam1')

        self.declare_parameter('left_topic', '/stereo/left/image_raw')
        self.declare_parameter('right_topic', '/stereo/right/image_raw')
        self.declare_parameter('imu_topic', '/imu/data_raw')
        self.declare_parameter('clock_topic', '/clock')

        self.declare_parameter('left_frame_id', 'cam0')
        self.declare_parameter('right_frame_id', 'cam1')
        self.declare_parameter('imu_frame_id', 'imu')

        # GT / reference trajectory
        self.declare_parameter('enable_gt', False)
        self.declare_parameter(
            'gt_csv',
            'pose/urban28-pankyo/global_pose.csv'
        )
        self.declare_parameter('gt_pose_topic', '/gt/pose')
        self.declare_parameter('gt_odom_topic', '/gt/odom')
        self.declare_parameter('gt_path_topic', '/gt/path')
        self.declare_parameter('gt_frame_id', 'map')
        self.declare_parameter('gt_child_frame_id', 'base_link')

        self.declare_parameter('playback_rate', 4.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('timer_period_sec', 0.002)

        self.dataset_root = self.get_parameter('dataset_root').value
        self.imu_source = str(self.get_parameter('imu_source').value).strip().lower()

        self.fog_use_dt_rate = bool(self.get_parameter('fog_use_dt_rate').value)
        self.fog_scale = float(self.get_parameter('fog_scale').value)
        self.fog_axis_permutation = self._parse_triplet_param(
            self.get_parameter('fog_axis_permutation').value,
            default=(0, 1, 2),
            allowed={0, 1, 2},
            name='fog_axis_permutation'
        )
        self.fog_axis_signs = self._parse_triplet_param(
            self.get_parameter('fog_axis_signs').value,
            default=(1, 1, 1),
            allowed={-1, 1},
            name='fog_axis_signs'
        )
        self.fog_auto_bias = bool(self.get_parameter('fog_auto_bias').value)
        self.fog_bias_estimation_duration_sec = float(
            self.get_parameter('fog_bias_estimation_duration_sec').value
        )

        self.enable_gt = bool(self.get_parameter('enable_gt').value)

        self.left_keywords = self._split_keywords(
            self.get_parameter('left_keywords').value
        )
        self.right_keywords = self._split_keywords(
            self.get_parameter('right_keywords').value
        )

        self.left_topic = self.get_parameter('left_topic').value
        self.right_topic = self.get_parameter('right_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.clock_topic = self.get_parameter('clock_topic').value

        self.left_frame_id = self.get_parameter('left_frame_id').value
        self.right_frame_id = self.get_parameter('right_frame_id').value
        self.imu_frame_id = self.get_parameter('imu_frame_id').value

        self.gt_pose_topic = self.get_parameter('gt_pose_topic').value
        self.gt_odom_topic = self.get_parameter('gt_odom_topic').value
        self.gt_path_topic = self.get_parameter('gt_path_topic').value
        self.gt_frame_id = self.get_parameter('gt_frame_id').value
        self.gt_child_frame_id = self.get_parameter('gt_child_frame_id').value

        self.playback_rate = float(self.get_parameter('playback_rate').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.timer_period_sec = float(self.get_parameter('timer_period_sec').value)

        stereo_stamp_param = self.get_parameter('stereo_stamp_csv').value
        imu_csv_param = self.get_parameter('imu_csv').value
        fog_csv_param = self.get_parameter('fog_csv').value
        image_root_param = self.get_parameter('image_root').value
        gt_csv_param = self.get_parameter('gt_csv').value

        self.stereo_stamp_csv = self._resolve_existing_file(
            stereo_stamp_param, fallback_name='stereo_stamp.csv'
        )
        self.imu_csv = self._resolve_existing_file(
            imu_csv_param, fallback_name='xsens_imu.csv'
        )
        self.image_root = self._resolve_existing_dir(
            image_root_param, fallback_dir_name='img'
        )

        self.fog_csv = None
        if self.imu_source == 'fog_xsens':
            self.fog_csv = self._resolve_existing_file(
                fog_csv_param, fallback_name='fog.csv'
            )

        self.gt_csv = None
        if self.enable_gt:
            self.gt_csv = self._resolve_existing_file(
                gt_csv_param, fallback_name='global_pose.csv'
            )

        if self.imu_source not in ('xsens', 'fog_xsens'):
            raise ValueError(
                f'Unsupported imu_source={self.imu_source}. '
                f'Use "xsens" or "fog_xsens".'
            )

        self.bridge = CvBridge()

        self.pub_left = self.create_publisher(Image, self.left_topic, 10)
        self.pub_right = self.create_publisher(Image, self.right_topic, 10)
        self.pub_imu = self.create_publisher(Imu, self.imu_topic, 200)
        self.pub_clock = self.create_publisher(Clock, self.clock_topic, 50)

        self.pub_gt_pose = None
        self.pub_gt_odom = None
        self.pub_gt_path = None

        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = self.gt_frame_id

        if self.enable_gt:
            self.pub_gt_pose = self.create_publisher(PoseStamped, self.gt_pose_topic, 50)
            self.pub_gt_odom = self.create_publisher(Odometry, self.gt_odom_topic, 50)
            self.pub_gt_path = self.create_publisher(Path, self.gt_path_topic, 10)

        self.get_logger().info(f'dataset_root = {self.dataset_root}')
        self.get_logger().info(f'image_root   = {self.image_root}')
        self.get_logger().info(f'stereo_csv   = {self.stereo_stamp_csv}')
        self.get_logger().info(f'imu_csv      = {self.imu_csv}')
        self.get_logger().info(f'imu_source   = {self.imu_source}')
        if self.fog_csv is not None:
            self.get_logger().info(f'fog_csv      = {self.fog_csv}')
        self.get_logger().info(f'enable_gt    = {self.enable_gt}')
        if self.gt_csv is not None:
            self.get_logger().info(f'gt_csv       = {self.gt_csv}')

        left_index, right_index = self._build_image_indices(self.image_root)
        self.get_logger().info(f'left images indexed  = {len(left_index)}')
        self.get_logger().info(f'right images indexed = {len(right_index)}')

        stereo_events = self._load_stereo_events(
            self.stereo_stamp_csv, left_index, right_index
        )
        imu_events = self._load_imu_events()
        gt_events: List[Event] = []
        if self.enable_gt and self.gt_csv is not None:
            gt_events = self._load_gt_events(self.gt_csv)

        self.events: List[Event] = sorted(
            stereo_events + imu_events + gt_events,
            key=lambda e: (e.ts_ns, 0 if e.kind == 'imu' else 1 if e.kind == 'gt' else 2)
        )

        if not self.events:
            raise RuntimeError('No events loaded. Check dataset_root / csv / img paths.')

        self.get_logger().info(f'total events loaded = {len(self.events)}')
        self.get_logger().info(f'first timestamp ns  = {self.events[0].ts_ns}')
        self.get_logger().info(f'last timestamp ns   = {self.events[-1].ts_ns}')

        self.idx = 0
        self.dataset_start_ns = self.events[0].ts_ns
        self.dataset_end_ns = self.events[-1].ts_ns
        self.wall_start_time = time.monotonic()

        self.timer = self.create_timer(self.timer_period_sec, self._on_timer)

    def _split_keywords(self, s: str) -> List[str]:
        return [x.strip().lower() for x in s.split(',') if x.strip()]

    def _parse_triplet_param(
        self,
        value,
        default: Tuple[int, int, int],
        allowed: set,
        name: str
    ) -> Tuple[int, int, int]:
        try:
            vals = tuple(int(x.strip()) for x in str(value).split(','))
            if len(vals) != 3:
                raise ValueError
            if any(v not in allowed for v in vals):
                raise ValueError
            return vals
        except Exception:
            self.get_logger().warning(
                f'Invalid {name}={value}, fallback to {default}'
            )
            return default

    def _robust_axis_center(self, values: List[float]) -> float:
        if not values:
            return 0.0

        med = statistics.median(values)
        abs_dev = [abs(v - med) for v in values]
        mad = statistics.median(abs_dev)

        # 幾乎沒有分散時，直接回 median
        if mad < 1e-12:
            return med

        sigma = 1.4826 * mad
        keep = [v for v in values if abs(v - med) <= 3.5 * sigma]

        if not keep:
            return med

        return sum(keep) / len(keep)


    def _interp_fog_sample(
        self,
        fog_samples: List[FogGyroSample],
        fog_ts: List[int],
        target_ts_ns: int
    ) -> Tuple[Optional[FogGyroSample], int]:
        """
        在 target_ts_ns 上對 FOG gyro 做線性插值。
        回傳:
        - FogGyroSample(ts=target_ts_ns, gx, gy, gz)
        - 與最近端點的時間差 abs dt (for logging only)
        """
        if not fog_samples:
            return None, 0

        idx = bisect_left(fog_ts, target_ts_ns)

        # target 在最前面
        if idx <= 0:
            fg = fog_samples[0]
            return FogGyroSample(
                ts_ns=target_ts_ns,
                gx=fg.gx,
                gy=fg.gy,
                gz=fg.gz
            ), abs(fg.ts_ns - target_ts_ns)

        # target 在最後面
        if idx >= len(fog_samples):
            fg = fog_samples[-1]
            return FogGyroSample(
                ts_ns=target_ts_ns,
                gx=fg.gx,
                gy=fg.gy,
                gz=fg.gz
            ), abs(fg.ts_ns - target_ts_ns)

        left = fog_samples[idx - 1]
        right = fog_samples[idx]

        dt_total = right.ts_ns - left.ts_ns
        if dt_total <= 0:
            # 防呆：退化時直接拿左邊
            return FogGyroSample(
                ts_ns=target_ts_ns,
                gx=left.gx,
                gy=left.gy,
                gz=left.gz
            ), min(abs(left.ts_ns - target_ts_ns), abs(right.ts_ns - target_ts_ns))

        alpha = (target_ts_ns - left.ts_ns) / float(dt_total)
        alpha = max(0.0, min(1.0, alpha))

        gx = (1.0 - alpha) * left.gx + alpha * right.gx
        gy = (1.0 - alpha) * left.gy + alpha * right.gy
        gz = (1.0 - alpha) * left.gz + alpha * right.gz

        dt_abs_ns = min(abs(left.ts_ns - target_ts_ns), abs(right.ts_ns - target_ts_ns))

        return FogGyroSample(
            ts_ns=target_ts_ns,
            gx=gx,
            gy=gy,
            gz=gz
        ), dt_abs_ns


    def _estimate_fog_bias_from_xsens(
        self,
        matched_pairs: List[Tuple[ImuSample, FogGyroSample, int]]
    ) -> Tuple[float, float, float]:
        """
        只用近似靜止樣本估計 FOG constant gyro bias。
        這裡估的是 player 端要扣掉的常數偏置 bx/by/bz，
        不是 VINS 裡的 bias random walk 參數。
        """
        if not matched_pairs:
            return 0.0, 0.0, 0.0

        t0 = matched_pairs[0][0].ts_ns

        # 可依資料再調
        gyro_static_thresh = 0.03   # rad/s
        acc_mag_thresh = 0.20       # m/s^2 around g
        g_ref = 9.81007

        diffs_x = []
        diffs_y = []
        diffs_z = []

        for xs, fg, _ in matched_pairs:
            if (xs.ts_ns - t0) * 1e-9 > self.fog_bias_estimation_duration_sec:
                break

            gyro_norm = math.sqrt(xs.gx * xs.gx + xs.gy * xs.gy + xs.gz * xs.gz)
            acc_norm = math.sqrt(xs.ax * xs.ax + xs.ay * xs.ay + xs.az * xs.az)

            # 只在近似靜止時拿來估 bias
            if gyro_norm < gyro_static_thresh and abs(acc_norm - g_ref) < acc_mag_thresh:
                diffs_x.append(fg.gx - xs.gx)
                diffs_y.append(fg.gy - xs.gy)
                diffs_z.append(fg.gz - xs.gz)

        if not diffs_x:
            self.get_logger().warning(
                'No quasi-static samples found for fog bias estimation; use zero bias.'
            )
            return 0.0, 0.0, 0.0

        bx = self._robust_axis_center(diffs_x)
        by = self._robust_axis_center(diffs_y)
        bz = self._robust_axis_center(diffs_z)

        return bx, by, bz


    def _merge_fog_xsens(
        self,
        xsens_samples: List[ImuSample],
        fog_samples: List[FogGyroSample]
    ) -> List[Event]:
        if not xsens_samples:
            return []

        if not fog_samples:
            raise RuntimeError('imu_source=fog_xsens but no fog samples were loaded.')

        fog_ts = [s.ts_ns for s in fog_samples]
        matched_pairs: List[Tuple[ImuSample, FogGyroSample, int]] = []

        for xs in xsens_samples:
            fg_interp, dt_abs_ns = self._interp_fog_sample(fog_samples, fog_ts, xs.ts_ns)
            if fg_interp is None:
                continue
            matched_pairs.append((xs, fg_interp, dt_abs_ns))

        if not matched_pairs:
            return []

        bx, by, bz = 0.0, 0.0, 0.0
        if self.fog_auto_bias:
            bx, by, bz = self._estimate_fog_bias_from_xsens(matched_pairs)
            self.get_logger().info(
                f'fog auto bias estimated from first '
                f'{self.fog_bias_estimation_duration_sec:.1f}s (quasi-static only): '
                f'[{bx:.9f}, {by:.9f}, {bz:.9f}]'
            )

        dt_abs_ns_list: List[int] = []
        events: List[Event] = []

        for xs, fg, dt_abs_ns in matched_pairs:
            dt_abs_ns_list.append(dt_abs_ns)

            merged = ImuSample(
                ts_ns=xs.ts_ns,
                ax=xs.ax,
                ay=xs.ay,
                az=xs.az,
                gx=fg.gx - bx,
                gy=fg.gy - by,
                gz=fg.gz - bz,
            )

            events.append(Event(ts_ns=merged.ts_ns, kind='imu', payload=merged))

        self.get_logger().info(f'fog_xsens imu events built = {len(events)}')

        if dt_abs_ns_list:
            avg_dt_ms = sum(dt_abs_ns_list) / len(dt_abs_ns_list) / 1e6
            max_dt_ms = max(dt_abs_ns_list) / 1e6
            self.get_logger().info(
                f'fog_xsens interp anchor dt: avg={avg_dt_ms:.3f} ms, max={max_dt_ms:.3f} ms'
            )

        return events

    def _resolve_path(self, path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        return os.path.join(self.dataset_root, path_str)

    def _auto_find_first_file(self, filename: str) -> Optional[str]:
        for dirpath, _, filenames in os.walk(self.dataset_root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
        return None

    def _auto_find_dir_by_name(self, dirname: str) -> Optional[str]:
        for dirpath, dirnames, _ in os.walk(self.dataset_root):
            if dirname in dirnames:
                return os.path.join(dirpath, dirname)
        return None

    def _resolve_existing_file(self, param_value: str, fallback_name: str) -> str:
        candidate = self._resolve_path(param_value)
        if os.path.isfile(candidate):
            return candidate

        found = self._auto_find_first_file(fallback_name)
        if found is not None:
            self.get_logger().warning(
                f'Configured path not found: {candidate}\n'
                f'Auto-discovered {fallback_name}: {found}'
            )
            return found

        raise FileNotFoundError(
            f'Cannot find file: configured="{candidate}", fallback_name="{fallback_name}"'
        )

    def _resolve_existing_dir(self, param_value: str, fallback_dir_name: str) -> str:
        candidate = self._resolve_path(param_value)
        if os.path.isdir(candidate):
            return candidate

        found = self._auto_find_dir_by_name(fallback_dir_name)
        if found is not None:
            self.get_logger().warning(
                f'Configured image_root not found: {candidate}\n'
                f'Auto-discovered {fallback_dir_name}: {found}'
            )
            return found

        raise FileNotFoundError(
            f'Cannot find directory: configured="{candidate}", fallback_dir_name="{fallback_dir_name}"'
        )

    def _build_image_indices(self, root_dir: str) -> Tuple[Dict[int, str], Dict[int, str]]:
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f'image_root does not exist: {root_dir}')

        left_index: Dict[int, str] = {}
        right_index: Dict[int, str] = {}

        for dirpath, _, filenames in os.walk(root_dir):
            path_lower = dirpath.lower()
            is_left = any(k in path_lower for k in self.left_keywords)
            is_right = any(k in path_lower for k in self.right_keywords)

            if not is_left and not is_right:
                continue

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue

                stem = os.path.splitext(fname)[0]
                ts_ns = parse_int_lossless(stem)
                if ts_ns is None:
                    continue

                full_path = os.path.join(dirpath, fname)

                if is_left and ts_ns not in left_index:
                    left_index[ts_ns] = full_path
                if is_right and ts_ns not in right_index:
                    right_index[ts_ns] = full_path

        if not left_index or not right_index:
            self.get_logger().error(
                'Failed to auto-detect left/right images.\n'
                'Please check img folder names and adjust left_keywords/right_keywords.'
            )

        return left_index, right_index

    def _extract_all_integer_timestamps(self, row: List[str]) -> List[int]:
        ts_list: List[int] = []
        for cell in row:
            ts = parse_int_lossless(cell)
            if ts is not None:
                ts_list.append(ts)
        return ts_list

    def _load_stereo_events(
        self,
        csv_path: str,
        left_index: Dict[int, str],
        right_index: Dict[int, str]
    ) -> List[Event]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'stereo_stamp.csv not found: {csv_path}')

        events: List[Event] = []
        missing_left = 0
        missing_right = 0
        unresolved_rows = 0

        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                nums = self._extract_all_integer_timestamps(row)
                if not nums:
                    continue

                ts_ns = nums[0]

                left_path = left_index.get(ts_ns)
                right_path = right_index.get(ts_ns)

                if left_path is None:
                    missing_left += 1
                    continue
                if right_path is None:
                    missing_right += 1
                    continue

                frame = StereoFrame(
                    ts_ns=ts_ns,
                    left_path=left_path,
                    right_path=right_path
                )
                events.append(Event(ts_ns=ts_ns, kind='stereo', payload=frame))

        self.get_logger().info(f'stereo frames loaded = {len(events)}')
        self.get_logger().info(f'missing left frames  = {missing_left}')
        self.get_logger().info(f'missing right frames = {missing_right}')
        self.get_logger().info(f'unresolved rows      = {unresolved_rows}')
        return events

    def _load_imu_events(self) -> List[Event]:
        xsens_samples = self._load_xsens_samples(self.imu_csv)

        if self.imu_source == 'xsens':
            self.get_logger().info('Building IMU events from xsens only.')
            return [Event(ts_ns=s.ts_ns, kind='imu', payload=s) for s in xsens_samples]

        if self.imu_source == 'fog_xsens':
            if self.fog_csv is None:
                raise RuntimeError('imu_source=fog_xsens but fog_csv is not available.')

            fog_samples = self._load_fog_samples(self.fog_csv)
            return self._merge_fog_xsens(xsens_samples, fog_samples)

        raise RuntimeError(f'Unsupported imu_source={self.imu_source}')

    def _load_xsens_samples(self, csv_path: str) -> List[ImuSample]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'xsens_imu.csv not found: {csv_path}')

        samples: List[ImuSample] = []

        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return samples

        header_map = None
        start_idx = 0

        if rows and any(not is_number(cell.strip()) for cell in rows[0] if cell.strip()):
            header_map = {normalize_header(name): idx for idx, name in enumerate(rows[0])}
            start_idx = 1

        for row in rows[start_idx:]:
            if not row:
                continue

            sample = self._parse_xsens_row(row, header_map)
            if sample is None:
                continue

            samples.append(sample)

        self.get_logger().info(f'xsens samples loaded = {len(samples)}')
        return samples

    def _load_fog_samples(self, csv_path: str) -> List[FogGyroSample]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'fog.csv not found: {csv_path}')

        raw_rows: List[Tuple[int, float, float, float]] = []

        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                try:
                    ts_ns = parse_int_lossless(row[0])
                    if ts_ns is None:
                        continue

                    rx = float(row[1].strip())
                    ry = float(row[2].strip())
                    rz = float(row[3].strip())

                    raw_rows.append((ts_ns, rx, ry, rz))
                except Exception:
                    continue

        if len(raw_rows) < 2:
            self.get_logger().warning('fog.csv has too few valid rows.')
            return []

        samples: List[FogGyroSample] = []
        dt_list_ms: List[float] = []

        for i, (ts_ns, rx, ry, rz) in enumerate(raw_rows):
            if self.fog_use_dt_rate:
                if i == 0:
                    dt_ns = raw_rows[i + 1][0] - ts_ns
                else:
                    dt_ns = ts_ns - raw_rows[i - 1][0]

                if dt_ns <= 0:
                    continue

                dt_sec = dt_ns * 1e-9
                dt_list_ms.append(dt_sec * 1e3)

                # 將 fog 原始量視為角度增量，換成角速度
                gx = rx / dt_sec
                gy = ry / dt_sec
                gz = rz / dt_sec
            else:
                gx = rx
                gy = ry
                gz = rz

            raw_vec = [gx, gy, gz]

            mapped = [
                self.fog_scale * self.fog_axis_signs[0] * raw_vec[self.fog_axis_permutation[0]],
                self.fog_scale * self.fog_axis_signs[1] * raw_vec[self.fog_axis_permutation[1]],
                self.fog_scale * self.fog_axis_signs[2] * raw_vec[self.fog_axis_permutation[2]],
            ]

            samples.append(
                FogGyroSample(
                    ts_ns=ts_ns,
                    gx=mapped[0],
                    gy=mapped[1],
                    gz=mapped[2],
                )
            )

        self.get_logger().info(f'fog gyro samples loaded = {len(samples)}')

        if dt_list_ms:
            avg_dt_ms = sum(dt_list_ms) / len(dt_list_ms)
            max_dt_ms = max(dt_list_ms)
            self.get_logger().info(
                f'fog dt used for rate conversion: avg={avg_dt_ms:.6f} ms, max={max_dt_ms:.6f} ms'
            )

        self.get_logger().info(
            f'fog config: use_dt_rate={self.fog_use_dt_rate}, '
            f'scale={self.fog_scale}, '
            f'perm={self.fog_axis_permutation}, '
            f'signs={self.fog_axis_signs}'
        )

        return samples

    def _load_gt_events(self, csv_path: str) -> List[Event]:
        samples = self._load_gt_samples(csv_path)
        events = [Event(ts_ns=s.ts_ns, kind='gt', payload=s) for s in samples]
        self.get_logger().info(f'gt pose samples loaded = {len(events)}')
        return events

    def _load_gt_samples(self, csv_path: str) -> List[PoseSample]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'global_pose.csv not found: {csv_path}')

        samples: List[PoseSample] = []

        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return samples

        header_map = None
        start_idx = 0

        if rows and any(not is_number(cell.strip()) for cell in rows[0] if cell.strip()):
            header_map = {normalize_header(name): idx for idx, name in enumerate(rows[0])}
            start_idx = 1

        for row in rows[start_idx:]:
            if not row:
                continue

            sample = self._parse_gt_row(row, header_map)
            if sample is None:
                continue

            samples.append(sample)

        return samples

    def _parse_gt_row(
        self,
        row: List[str],
        header_map: Optional[Dict[str, int]]
    ) -> Optional[PoseSample]:
        if header_map is not None:
            sample = self._parse_gt_row_with_header(row, header_map)
            if sample is not None:
                return sample

        raw_cells = [c.strip() for c in row if c.strip()]
        vals: List[float] = []
        for c in raw_cells:
            if is_number(c):
                vals.append(float(c))

        # timestamp + 3x4 pose matrix
        if len(vals) >= 13:
            ts_ns = parse_int_lossless(raw_cells[0])
            if ts_ns is None:
                return None

            r00, r01, r02, tx = vals[1], vals[2], vals[3], vals[4]
            r10, r11, r12, ty = vals[5], vals[6], vals[7], vals[8]
            r20, r21, r22, tz = vals[9], vals[10], vals[11], vals[12]

            qx, qy, qz, qw = quaternion_from_rotation_matrix(
                r00, r01, r02,
                r10, r11, r12,
                r20, r21, r22
            )
            return PoseSample(ts_ns, tx, ty, tz, qx, qy, qz, qw)

        # timestamp + tx ty tz qx qy qz qw
        if len(vals) >= 8:
            ts_ns = parse_int_lossless(raw_cells[0])
            if ts_ns is None:
                return None

            tx, ty, tz = vals[1], vals[2], vals[3]
            qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
            return PoseSample(ts_ns, tx, ty, tz, qx, qy, qz, qw)

        return None

    def _parse_gt_row_with_header(
        self,
        row: List[str],
        h: Dict[str, int]
    ) -> Optional[PoseSample]:
        def find_idx(candidates: List[str]) -> Optional[int]:
            for c in candidates:
                if c in h:
                    return h[c]
            return None

        idx_t = find_idx(['timestamp', 'stamp', 'time', 'time_ns', 'timestamp_ns', 'ts'])

        idx_tx = find_idx(['tx', 'x', 'px', 'pos_x', 'position_x'])
        idx_ty = find_idx(['ty', 'y', 'py', 'pos_y', 'position_y'])
        idx_tz = find_idx(['tz', 'z', 'pz', 'pos_z', 'position_z'])

        idx_qx = find_idx(['qx'])
        idx_qy = find_idx(['qy'])
        idx_qz = find_idx(['qz'])
        idx_qw = find_idx(['qw'])

        if all(v is not None for v in [idx_t, idx_tx, idx_ty, idx_tz, idx_qx, idx_qy, idx_qz, idx_qw]):
            try:
                ts_ns = parse_int_lossless(row[idx_t])
                if ts_ns is None:
                    return None

                tx = float(row[idx_tx])
                ty = float(row[idx_ty])
                tz = float(row[idx_tz])
                qx = float(row[idx_qx])
                qy = float(row[idx_qy])
                qz = float(row[idx_qz])
                qw = float(row[idx_qw])

                return PoseSample(ts_ns, tx, ty, tz, qx, qy, qz, qw)
            except Exception:
                return None

        idx_r00 = find_idx(['r00'])
        idx_r01 = find_idx(['r01'])
        idx_r02 = find_idx(['r02'])
        idx_r10 = find_idx(['r10'])
        idx_r11 = find_idx(['r11'])
        idx_r12 = find_idx(['r12'])
        idx_r20 = find_idx(['r20'])
        idx_r21 = find_idx(['r21'])
        idx_r22 = find_idx(['r22'])

        if all(v is not None for v in [idx_t, idx_tx, idx_ty, idx_tz, idx_r00, idx_r01, idx_r02, idx_r10, idx_r11, idx_r12, idx_r20, idx_r21, idx_r22]):
            try:
                ts_ns = parse_int_lossless(row[idx_t])
                if ts_ns is None:
                    return None

                tx = float(row[idx_tx])
                ty = float(row[idx_ty])
                tz = float(row[idx_tz])

                qx, qy, qz, qw = quaternion_from_rotation_matrix(
                    float(row[idx_r00]), float(row[idx_r01]), float(row[idx_r02]),
                    float(row[idx_r10]), float(row[idx_r11]), float(row[idx_r12]),
                    float(row[idx_r20]), float(row[idx_r21]), float(row[idx_r22]),
                )
                return PoseSample(ts_ns, tx, ty, tz, qx, qy, qz, qw)
            except Exception:
                return None

        return None

    def _parse_xsens_row(
        self,
        row: List[str],
        header_map: Optional[Dict[str, int]]
    ) -> Optional[ImuSample]:
        if header_map is not None:
            sample = self._parse_xsens_row_with_header(row, header_map)
            if sample is not None:
                return sample

        # Observed KAIST xsens_imu.csv format (17 columns):
        # [0] timestamp
        # [8:11] gyro xyz
        # [11:14] accel xyz
        if len(row) >= 17:
            try:
                ts_ns = parse_int_lossless(row[0])
                if ts_ns is None:
                    return None

                gx = float(row[8].strip())
                gy = float(row[9].strip())
                gz = float(row[10].strip())

                ax = float(row[11].strip())
                ay = float(row[12].strip())
                az = float(row[13].strip())

                return ImuSample(ts_ns, ax, ay, az, gx, gy, gz)
            except Exception:
                pass

        nums_str = [c.strip() for c in row if c.strip()]
        nums = []
        for cell in nums_str:
            if is_number(cell):
                nums.append(float(cell))

        if len(nums) >= 11:
            ts_ns = parse_int_lossless(nums_str[0])
            if ts_ns is None:
                return None
            ax, ay, az = nums[5], nums[6], nums[7]
            gx, gy, gz = nums[8], nums[9], nums[10]
            return ImuSample(ts_ns, ax, ay, az, gx, gy, gz)

        if len(nums) >= 7:
            ts_ns = parse_int_lossless(nums_str[0])
            if ts_ns is None:
                return None
            ax, ay, az = nums[1], nums[2], nums[3]
            gx, gy, gz = nums[4], nums[5], nums[6]
            return ImuSample(ts_ns, ax, ay, az, gx, gy, gz)

        return None

    def _parse_xsens_row_with_header(
        self,
        row: List[str],
        h: Dict[str, int]
    ) -> Optional[ImuSample]:
        def find_idx(candidates: List[str]) -> Optional[int]:
            for c in candidates:
                if c in h:
                    return h[c]
            return None

        idx_t = find_idx(['timestamp', 'stamp', 'time', 'time_ns', 'timestamp_ns', 'ts'])
        idx_ax = find_idx(['ax', 'acc_x', 'accel_x', 'linear_acceleration_x', 'acceleration_x'])
        idx_ay = find_idx(['ay', 'acc_y', 'accel_y', 'linear_acceleration_y', 'acceleration_y'])
        idx_az = find_idx(['az', 'acc_z', 'accel_z', 'linear_acceleration_z', 'acceleration_z'])

        idx_gx = find_idx(['gx', 'wx', 'gyro_x', 'angular_velocity_x'])
        idx_gy = find_idx(['gy', 'wy', 'gyro_y', 'angular_velocity_y'])
        idx_gz = find_idx(['gz', 'wz', 'gyro_z', 'angular_velocity_z'])

        required = [idx_t, idx_ax, idx_ay, idx_az, idx_gx, idx_gy, idx_gz]
        if any(x is None for x in required):
            return None

        try:
            ts_ns = parse_int_lossless(row[idx_t])
            if ts_ns is None:
                return None

            ax = float(row[idx_ax])
            ay = float(row[idx_ay])
            az = float(row[idx_az])
            gx = float(row[idx_gx])
            gy = float(row[idx_gy])
            gz = float(row[idx_gz])

            return ImuSample(ts_ns, ax, ay, az, gx, gy, gz)
        except Exception:
            return None

    def _publish_clock(self, ts_ns: int) -> None:
        sec, nsec = sec_nsec_from_ns(ts_ns)
        msg = Clock()
        msg.clock.sec = sec
        msg.clock.nanosec = nsec
        self.pub_clock.publish(msg)

    def _make_image_msg(self, img, ts_ns: int, frame_id: str) -> Image:
        if len(img.shape) == 2:
            encoding = 'mono8'
        elif len(img.shape) == 3 and img.shape[2] == 3:
            encoding = 'bgr8'
        elif len(img.shape) == 3 and img.shape[2] == 4:
            encoding = 'bgra8'
        else:
            encoding = 'passthrough'

        msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
        sec, nsec = sec_nsec_from_ns(ts_ns)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = frame_id
        return msg

    def _publish_stereo(self, frame: StereoFrame) -> None:
        left_img = cv2.imread(frame.left_path, cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(frame.right_path, cv2.IMREAD_UNCHANGED)

        if left_img is None:
            self.get_logger().warning(f'Failed to read left image: {frame.left_path}')
            return
        if right_img is None:
            self.get_logger().warning(f'Failed to read right image: {frame.right_path}')
            return

        left_msg = self._make_image_msg(left_img, frame.ts_ns, self.left_frame_id)
        right_msg = self._make_image_msg(right_img, frame.ts_ns, self.right_frame_id)

        self.pub_left.publish(left_msg)
        self.pub_right.publish(right_msg)

    def _publish_imu(self, sample: ImuSample) -> None:
        msg = Imu()
        sec, nsec = sec_nsec_from_ns(sample.ts_ns)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.header.frame_id = self.imu_frame_id

        msg.orientation_covariance[0] = -1.0

        msg.angular_velocity.x = sample.gx
        msg.angular_velocity.y = sample.gy
        msg.angular_velocity.z = sample.gz

        msg.linear_acceleration.x = sample.ax
        msg.linear_acceleration.y = sample.ay
        msg.linear_acceleration.z = sample.az

        self.pub_imu.publish(msg)

    def _publish_gt(self, sample: PoseSample) -> None:
        if self.pub_gt_pose is None or self.pub_gt_odom is None or self.pub_gt_path is None:
            return

        sec, nsec = sec_nsec_from_ns(sample.ts_ns)

        pose_msg = PoseStamped()
        pose_msg.header.stamp.sec = sec
        pose_msg.header.stamp.nanosec = nsec
        pose_msg.header.frame_id = self.gt_frame_id
        pose_msg.pose.position.x = sample.tx
        pose_msg.pose.position.y = sample.ty
        pose_msg.pose.position.z = sample.tz
        pose_msg.pose.orientation.x = sample.qx
        pose_msg.pose.orientation.y = sample.qy
        pose_msg.pose.orientation.z = sample.qz
        pose_msg.pose.orientation.w = sample.qw

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = self.gt_child_frame_id
        odom_msg.pose.pose = pose_msg.pose

        self.pub_gt_pose.publish(pose_msg)
        self.pub_gt_odom.publish(odom_msg)

        self.gt_path_msg.header.stamp = pose_msg.header.stamp
        self.gt_path_msg.poses.append(pose_msg)
        self.pub_gt_path.publish(self.gt_path_msg)

    def _reset_playback(self) -> None:
        self.idx = 0
        self.wall_start_time = time.monotonic()
        self.dataset_start_ns = self.events[0].ts_ns
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = self.gt_frame_id
        self.get_logger().info('Looping playback from beginning.')

    def _on_timer(self) -> None:
        if self.idx >= len(self.events):
            if self.loop:
                self._reset_playback()
            return

        elapsed_wall = time.monotonic() - self.wall_start_time
        sim_now_ns = self.dataset_start_ns + int(elapsed_wall * self.playback_rate * 1e9)

        self._publish_clock(sim_now_ns)

        while self.idx < len(self.events) and self.events[self.idx].ts_ns <= sim_now_ns:
            ev = self.events[self.idx]

            self._publish_clock(ev.ts_ns)

            if ev.kind == 'imu':
                self._publish_imu(ev.payload)
            elif ev.kind == 'gt':
                self._publish_gt(ev.payload)
            elif ev.kind == 'stereo':
                self._publish_stereo(ev.payload)

            self.idx += 1

        if self.idx == len(self.events):
            self.get_logger().info('Playback finished.')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = KaistPlayerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'[kaist_player_node] Fatal error: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
