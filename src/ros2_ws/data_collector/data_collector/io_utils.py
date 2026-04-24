import csv
import os
from pathlib import Path

import cv2


IMU_HEADER = [
    'id',
    'timestamp_sec', 'timestamp_nsec',
    'ori_x', 'ori_y', 'ori_z', 'ori_w',
    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
]

ODOM_HEADER = [
    'odom_id',
    'timestamp_sec', 'timestamp_nsec',
    'pos_x', 'pos_y', 'pos_z',
    'quat_x', 'quat_y', 'quat_z', 'quat_w',
]

CLOCK_HEADER = [
    'clock_id',
    'clock_sec', 'clock_nsec',
]

FRAME_INDEX_HEADER = [
    'source_frame_id',
    'image_file',
    'timestamp_token',
    'image_stamp_sec', 'image_stamp_nsec',
    'clock_sec', 'clock_nsec',
    'imu_start_id', 'imu_end_id',
    'fog_start_id', 'fog_end_id',
    'odom_stamp_sec', 'odom_stamp_nsec',
    'pos_x', 'pos_y', 'pos_z',
    'quat_x', 'quat_y', 'quat_z', 'quat_w',
    'yaw_deg',
]

GC_LOG_HEADER = [
    'source_frame_id',
    'image_file',
    'timestamp_token',
    'reason',
]

VEHICLE_POSE_HEADER = [
    'timestamp_sec', 'timestamp_nsec',
    'P00', 'P01', 'P02', 'P03',
    'P10', 'P11', 'P12', 'P13',
    'P20', 'P21', 'P22', 'P23',
]


class DatasetLayout:
    def __init__(self, output_root: str, sequence_name: str):
        root = Path(os.path.expanduser(output_root))
        self.sequence_dir = root / sequence_name

        self.images_dir = self.sequence_dir / 'images'
        self.imu_csv = self.sequence_dir / 'imu.csv'
        self.pseudo_fog_csv = self.sequence_dir / 'pseudo_fog.csv'
        self.frame_index_csv = self.sequence_dir / 'frame_index.csv'
        self.buffer_manifest_csv = self.sequence_dir / 'buffer_manifest.csv'
        self.gc_log_csv = self.sequence_dir / 'gc_log.csv'
        self.camera_info_yaml = self.sequence_dir / 'camera_info.yaml'
        self.metadata_yaml = self.sequence_dir / 'metadata.yaml'

        self.sensor_data_dir = self.sequence_dir / 'sensor_data'
        self.raw_odom_csv = self.sensor_data_dir / 'odom.csv'
        self.raw_clock_csv = self.sensor_data_dir / 'clock.csv'

        self.calibration_dir = self.sequence_dir / 'calibration'
        self.calibration_camera_info_yaml = self.calibration_dir / 'camera_info_front.yaml'

        self.baseline_dir = self.sequence_dir / 'baseline'
        self.vehicle_pose_csv = self.baseline_dir / 'vehicle_pose.csv'

        self.keyframes_dir = self.sequence_dir / 'keyframes'
        self.keyframe_images_dir = self.keyframes_dir / 'images'
        self.keyframes_csv = self.keyframes_dir / 'keyframes.csv'

    def create_dirs(self):
        dirs = [
            self.sequence_dir,
            self.images_dir,
            self.sensor_data_dir,
            self.calibration_dir,
            self.baseline_dir,
            self.keyframes_dir,
            self.keyframe_images_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def open_csv_writer(path: Path, header):
    f = open(path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(header)
    f.flush()
    return f, writer


def save_png(path: Path, image):
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f'Failed to save image: {path}')


def _camera_info_yaml_lines(msg):
    lines = []
    lines.append(f'frame_id: "{msg.header.frame_id}"')
    lines.append(f'width: {msg.width}')
    lines.append(f'height: {msg.height}')
    lines.append(f'distortion_model: "{msg.distortion_model}"')
    lines.append(f'd: {list(msg.d)}')
    lines.append(f'k: {list(msg.k)}')
    lines.append(f'r: {list(msg.r)}')
    lines.append(f'p: {list(msg.p)}')
    return lines


def write_camera_info_yaml(path: Path, msg):
    lines = _camera_info_yaml_lines(msg)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def write_metadata_yaml(path: Path, metadata: dict):
    lines = []
    for key, value in metadata.items():
        if isinstance(value, bool):
            value_str = 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            value_str = str(value)
        else:
            value_str = f'"{value}"'
        lines.append(f'{key}: {value_str}')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def rewrite_buffer_manifest(path: Path, rows):
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_INDEX_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


class CollectorWriters:
    def __init__(self, layout: DatasetLayout):
        self.layout = layout
        self.layout.create_dirs()

        self.imu_file, self.imu_writer = open_csv_writer(
            self.layout.imu_csv, IMU_HEADER
        )
        self.pseudo_fog_file, self.pseudo_fog_writer = open_csv_writer(
            self.layout.pseudo_fog_csv, IMU_HEADER
        )
        self.frame_index_file, self.frame_index_writer = open_csv_writer(
            self.layout.frame_index_csv, FRAME_INDEX_HEADER
        )
        self.gc_log_file, self.gc_log_writer = open_csv_writer(
            self.layout.gc_log_csv, GC_LOG_HEADER
        )
        self.raw_odom_file, self.raw_odom_writer = open_csv_writer(
            self.layout.raw_odom_csv, ODOM_HEADER
        )
        self.raw_clock_file, self.raw_clock_writer = open_csv_writer(
            self.layout.raw_clock_csv, CLOCK_HEADER
        )
        self.vehicle_pose_file, self.vehicle_pose_writer = open_csv_writer(
            self.layout.vehicle_pose_csv, VEHICLE_POSE_HEADER
        )

    def write_internal_imu(self, row):
        self.imu_writer.writerow(row)
        self.imu_file.flush()

    def write_internal_pseudo_fog(self, row):
        self.pseudo_fog_writer.writerow(row)
        self.pseudo_fog_file.flush()

    def write_frame_index(self, row_dict):
        self.frame_index_writer.writerow([row_dict[k] for k in FRAME_INDEX_HEADER])
        self.frame_index_file.flush()

    def write_gc_log(self, row_dict):
        self.gc_log_writer.writerow([row_dict[k] for k in GC_LOG_HEADER])
        self.gc_log_file.flush()

    def write_raw_odom(self, row):
        self.raw_odom_writer.writerow(row)
        self.raw_odom_file.flush()

    def write_raw_clock(self, row):
        self.raw_clock_writer.writerow(row)
        self.raw_clock_file.flush()

    def write_vehicle_pose(self, row):
        self.vehicle_pose_writer.writerow(row)
        self.vehicle_pose_file.flush()

    def close(self):
        files = [
            self.imu_file,
            self.pseudo_fog_file,
            self.frame_index_file,
            self.gc_log_file,
            self.raw_odom_file,
            self.raw_clock_file,
            self.vehicle_pose_file,
        ]
        for f in files:
            try:
                f.close()
            except Exception:
                pass