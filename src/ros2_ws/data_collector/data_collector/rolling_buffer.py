from collections import deque
from pathlib import Path


class RollingImageBuffer:
    def __init__(self, max_images: int):
        self.max_images = int(max_images)
        self.entries = deque()

    def push(self, entry: dict):
        """
        entry 至少要有：
        - image_path
        - 其他 manifest 需要的欄位
        """
        self.entries.append(entry)

        evicted = None
        if len(self.entries) > self.max_images:
            evicted = self.entries.popleft()
            try:
                Path(evicted['image_path']).unlink(missing_ok=True)
            except Exception:
                pass

        return evicted

    def manifest_rows(self):
        rows = []
        for e in self.entries:
            rows.append({
                'source_frame_id': e['source_frame_id'],
                'image_file': e['image_file'],
                'timestamp_token': e['timestamp_token'],
                'image_stamp_sec': e['image_stamp_sec'],
                'image_stamp_nsec': e['image_stamp_nsec'],
                'clock_sec': e['clock_sec'],
                'clock_nsec': e['clock_nsec'],
                'imu_start_id': e['imu_start_id'],
                'imu_end_id': e['imu_end_id'],
                'fog_start_id': e['fog_start_id'],
                'fog_end_id': e['fog_end_id'],
                'odom_stamp_sec': e['odom_stamp_sec'],
                'odom_stamp_nsec': e['odom_stamp_nsec'],
                'pos_x': e['pos_x'],
                'pos_y': e['pos_y'],
                'pos_z': e['pos_z'],
                'quat_x': e['quat_x'],
                'quat_y': e['quat_y'],
                'quat_z': e['quat_z'],
                'quat_w': e['quat_w'],
                'yaw_deg': e['yaw_deg'],
            })
        return rows