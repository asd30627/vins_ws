#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, Imu, CameraInfo
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from .format_utils import (
    stamp_to_filename,
    pose_to_matrix_row,
    quaternion_to_yaw_deg,
)
from .io_utils import (
    DatasetLayout,
    CollectorWriters,
    save_png,
    write_camera_info_yaml,
    write_metadata_yaml,
    rewrite_buffer_manifest,
)
from .rolling_buffer import RollingImageBuffer


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.declare_parameter('image_topic', '/camera/rgb/front')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('fog_topic', '/pseudo_fog/imu')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('camera_info_topic', '/camera_info/front')
        self.declare_parameter('clock_topic', '/clock')

        self.declare_parameter('output_root', '/mnt/sata4t/dataset/')
        self.declare_parameter('sequence_name', 'sequence_000')
        self.declare_parameter('max_images', 3000)

        self.declare_parameter('queue_size', 30)
        self.declare_parameter('sync_slop', 0.05)

        self.image_topic = self.get_parameter('image_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.fog_topic = self.get_parameter('fog_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.clock_topic = self.get_parameter('clock_topic').value

        self.output_root = os.path.expanduser(self.get_parameter('output_root').value)
        self.sequence_name = self.get_parameter('sequence_name').value
        self.max_images = int(self.get_parameter('max_images').value)

        self.queue_size = int(self.get_parameter('queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)

        self.layout = DatasetLayout(self.output_root, self.sequence_name)
        self.writers = CollectorWriters(self.layout)
        self.buffer = RollingImageBuffer(self.max_images)

        write_metadata_yaml(
            self.layout.metadata_yaml,
            {
                'schema_version': 'collector_v5_rolling_3000_then_select_keyframes',
                'sequence_name': self.sequence_name,
                'image_topic': self.image_topic,
                'imu_topic': self.imu_topic,
                'fog_topic': self.fog_topic,
                'odom_topic': self.odom_topic,
                'camera_info_topic': self.camera_info_topic,
                'clock_topic': self.clock_topic,
                'max_images': self.max_images,
                'queue_size': self.queue_size,
                'sync_slop': self.sync_slop,
            }
        )

        self.bridge = CvBridge()

        self.source_frame_id_counter = 0
        self.imu_id_counter = 0
        self.fog_id_counter = 0
        self.odom_id_counter = 0
        self.clock_id_counter = 0

        self.next_imu_start_id = 0
        self.next_fog_start_id = 0

        self.latest_clock_sec = None
        self.latest_clock_nsec = None

        self.camera_info_saved = False

        self.clock_sub = self.create_subscription(
            Clock,
            self.clock_topic,
            self.clock_callback,
            qos_profile_sensor_data
        )

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            qos_profile_sensor_data
        )

        self.fog_sub = self.create_subscription(
            Imu,
            self.fog_topic,
            self.fog_callback,
            qos_profile_sensor_data
        )

        self.odom_raw_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_raw_callback,
            qos_profile_sensor_data
        )

        self.image_sub = Subscriber(
            self,
            Image,
            self.image_topic,
            qos_profile=qos_profile_sensor_data
        )
        self.odom_sync_sub = Subscriber(
            self,
            Odometry,
            self.odom_topic,
            qos_profile=qos_profile_sensor_data
        )
        self.camera_info_sub = Subscriber(
            self,
            CameraInfo,
            self.camera_info_topic,
            qos_profile=qos_profile_sensor_data
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.odom_sync_sub, self.camera_info_sub],
            queue_size=self.queue_size,
            slop=self.sync_slop
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info('=== Data Collector Started ===')
        self.get_logger().info(f'image topic       : {self.image_topic}')
        self.get_logger().info(f'imu topic         : {self.imu_topic}')
        self.get_logger().info(f'fog topic         : {self.fog_topic}')
        self.get_logger().info(f'odom topic        : {self.odom_topic}')
        self.get_logger().info(f'clock topic       : {self.clock_topic}')
        self.get_logger().info(f'output dir        : {self.layout.sequence_dir}')
        self.get_logger().info(f'max_images        : {self.max_images}')

    def clock_callback(self, msg: Clock):
        sec = msg.clock.sec
        nsec = msg.clock.nanosec

        self.latest_clock_sec = sec
        self.latest_clock_nsec = nsec

        self.writers.write_raw_clock([
            self.clock_id_counter,
            sec,
            nsec,
        ])
        self.clock_id_counter += 1

    def imu_callback(self, msg: Imu):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec

        row = [
            self.imu_id_counter,
            sec,
            nsec,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]
        self.writers.write_internal_imu(row)
        self.imu_id_counter += 1

    def fog_callback(self, msg: Imu):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec

        row = [
            self.fog_id_counter,
            sec,
            nsec,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]
        self.writers.write_internal_pseudo_fog(row)
        self.fog_id_counter += 1

    def odom_raw_callback(self, msg: Odometry):
        sec = msg.header.stamp.sec
        nsec = msg.header.stamp.nanosec

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.writers.write_raw_odom([
            self.odom_id_counter,
            sec,
            nsec,
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
        ])

        mat_row = pose_to_matrix_row(
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w
        )
        self.writers.write_vehicle_pose([
            sec,
            nsec,
            *mat_row
        ])

        self.odom_id_counter += 1

    def synced_callback(self, image_msg: Image, odom_msg: Odometry, camera_info_msg: CameraInfo):
        source_frame_id = self.source_frame_id_counter

        image_sec = image_msg.header.stamp.sec
        image_nsec = image_msg.header.stamp.nanosec
        timestamp_token = stamp_to_filename(image_sec, image_nsec)

        image_file = f'{source_frame_id:06d}.png'
        image_path = self.layout.images_dir / image_file

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion failed: {e}')
            self.source_frame_id_counter += 1
            return

        try:
            save_png(image_path, cv_image)
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')
            self.source_frame_id_counter += 1
            return

        if not self.camera_info_saved:
            write_camera_info_yaml(self.layout.camera_info_yaml, camera_info_msg)
            write_camera_info_yaml(self.layout.calibration_camera_info_yaml, camera_info_msg)
            self.camera_info_saved = True
            self.get_logger().info('Saved camera_info files.')

        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        yaw_deg = quaternion_to_yaw_deg(q.x, q.y, q.z, q.w)

        imu_start_id = self.next_imu_start_id
        imu_end_id = self.imu_id_counter - 1
        if imu_end_id < imu_start_id:
            imu_start_id = -1
            imu_end_id = -1

        fog_start_id = self.next_fog_start_id
        fog_end_id = self.fog_id_counter - 1
        if fog_end_id < fog_start_id:
            fog_start_id = -1
            fog_end_id = -1

        if self.latest_clock_sec is None or self.latest_clock_nsec is None:
            clock_sec = image_sec
            clock_nsec = image_nsec
        else:
            clock_sec = self.latest_clock_sec
            clock_nsec = self.latest_clock_nsec

        row = {
            'source_frame_id': source_frame_id,
            'image_file': image_file,
            'timestamp_token': timestamp_token,
            'image_stamp_sec': image_sec,
            'image_stamp_nsec': image_nsec,
            'clock_sec': clock_sec,
            'clock_nsec': clock_nsec,
            'imu_start_id': imu_start_id,
            'imu_end_id': imu_end_id,
            'fog_start_id': fog_start_id,
            'fog_end_id': fog_end_id,
            'odom_stamp_sec': odom_msg.header.stamp.sec,
            'odom_stamp_nsec': odom_msg.header.stamp.nanosec,
            'pos_x': p.x,
            'pos_y': p.y,
            'pos_z': p.z,
            'quat_x': q.x,
            'quat_y': q.y,
            'quat_z': q.z,
            'quat_w': q.w,
            'yaw_deg': yaw_deg,
        }

        self.writers.write_frame_index(row)

        buffer_entry = dict(row)
        buffer_entry['image_path'] = str(image_path)
        evicted = self.buffer.push(buffer_entry)

        if evicted is not None:
            self.writers.write_gc_log({
                'source_frame_id': evicted['source_frame_id'],
                'image_file': evicted['image_file'],
                'timestamp_token': evicted['timestamp_token'],
                'reason': 'rolling_buffer_over_max',
            })

        rewrite_buffer_manifest(
            self.layout.buffer_manifest_csv,
            self.buffer.manifest_rows()
        )

        self.get_logger().info(
            f'frame={source_frame_id:06d} | image={image_file} | '
            f'imu=[{imu_start_id},{imu_end_id}] | fog=[{fog_start_id},{fog_end_id}] | '
            f'buffer_size={len(self.buffer.entries)}'
        )

        self.next_imu_start_id = self.imu_id_counter
        self.next_fog_start_id = self.fog_id_counter
        self.source_frame_id_counter += 1

    def destroy_node(self):
        try:
            self.writers.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()