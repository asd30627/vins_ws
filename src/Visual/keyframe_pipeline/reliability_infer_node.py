#!/usr/bin/env python3
import json
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import torch

from train_reliability_model import build_model


RUNTIME_FEATURE_NAMES = [
    'tracked_feature_count_raw',
    'tracked_feature_count_mgr',
    'current_is_keyframe',
    'outlier_count_last',
    'inlier_count_last',
    'outlier_ratio_last',
    'solver_time_ms_last',
    'failure_detected_last',
]


class ReliabilityInferNode(Node):
    def __init__(self):
        super().__init__('reliability_infer_node')

        self.declare_parameter('feature_topic', '/vins_admission/features')
        self.declare_parameter('prediction_topic', '/vins_admission/prediction')
        self.declare_parameter('use_dummy_prediction', True)
        self.declare_parameter('tau', 0.5)
        self.declare_parameter('seq_len', 1)
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('stats_json', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('hidden_dim', 64)
        self.declare_parameter('num_layers', 1)
        self.declare_parameter('dropout', 0.1)

        self.feature_topic = self.get_parameter('feature_topic').get_parameter_value().string_value
        self.prediction_topic = self.get_parameter('prediction_topic').get_parameter_value().string_value
        self.use_dummy_prediction = self.get_parameter('use_dummy_prediction').get_parameter_value().bool_value
        self.tau = float(self.get_parameter('tau').value)
        self.seq_len = int(self.get_parameter('seq_len').value)
        self.checkpoint_path = self.get_parameter('checkpoint').get_parameter_value().string_value
        self.stats_json_path = self.get_parameter('stats_json').get_parameter_value().string_value
        self.device = torch.device(self.get_parameter('device').get_parameter_value().string_value)
        self.hidden_dim = int(self.get_parameter('hidden_dim').value)
        self.num_layers = int(self.get_parameter('num_layers').value)
        self.dropout = float(self.get_parameter('dropout').value)

        self.publisher_ = self.create_publisher(Float64MultiArray, self.prediction_topic, 1)
        self.subscription = self.create_subscription(Float64MultiArray, self.feature_topic, self.feature_callback, 1)

        self.feature_buffer = deque(maxlen=max(self.seq_len, 1))
        self.last_stamp = -1.0

        self.model = None
        self.fill_values = {name: 0.0 for name in RUNTIME_FEATURE_NAMES}
        self.standardize_mean = np.zeros(len(RUNTIME_FEATURE_NAMES), dtype=np.float32)
        self.standardize_std = np.ones(len(RUNTIME_FEATURE_NAMES), dtype=np.float32)

        if not self.use_dummy_prediction:
            self._load_model_and_stats()
        else:
            self.get_logger().info('use_dummy_prediction=True, first version uses heuristic prediction')

    def _load_model_and_stats(self):
        if not self.checkpoint_path or not self.stats_json_path:
            raise RuntimeError('use_dummy_prediction=False 時，必須提供 checkpoint 與 stats_json')

        stats_path = Path(self.stats_json_path).expanduser().resolve()
        ckpt_path = Path(self.checkpoint_path).expanduser().resolve()

        if not stats_path.exists():
            raise FileNotFoundError(f'找不到 stats_json: {stats_path}')
        if not ckpt_path.exists():
            raise FileNotFoundError(f'找不到 checkpoint: {ckpt_path}')

        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        stats_feature_names = stats.get('feature_names', [])
        if list(stats_feature_names) != list(RUNTIME_FEATURE_NAMES):
            raise RuntimeError(
                '目前 runtime feature schema 與 stats_json 不一致。\n'
                f'expected={RUNTIME_FEATURE_NAMES}\n'
                f'got={stats_feature_names}'
            )

        self.fill_values = {k: float(v) for k, v in stats.get('fill_values', {}).items()}
        self.standardize_mean = np.asarray(stats.get('standardize_mean', [0.0] * len(RUNTIME_FEATURE_NAMES)), dtype=np.float32)
        self.standardize_std = np.asarray(stats.get('standardize_std', [1.0] * len(RUNTIME_FEATURE_NAMES)), dtype=np.float32)
        self.standardize_std[self.standardize_std < 1e-6] = 1.0

        ckpt = torch.load(ckpt_path, map_location=self.device)
        model_type = ckpt.get('model_type', 'mlp')
        seq_len_ckpt = int(ckpt.get('seq_len', self.seq_len))
        input_dim = int(ckpt.get('input_dim', len(RUNTIME_FEATURE_NAMES)))
        hidden_dim = int(ckpt.get('hidden_dim', self.hidden_dim))
        num_layers = int(ckpt.get('num_layers', self.num_layers))
        dropout = float(ckpt.get('dropout', self.dropout))

        self.seq_len = seq_len_ckpt
        self.feature_buffer = deque(maxlen=max(self.seq_len, 1))

        self.model = build_model(
            model_type=model_type,
            seq_len=self.seq_len,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f'loaded model: {ckpt_path}, model_type={model_type}, seq_len={self.seq_len}')

    def publish_prediction(self, stamp_sec: float, w_pred: float, soft_target_proxy: float):
        msg = Float64MultiArray()
        msg.data = [float(stamp_sec), float(w_pred), float(soft_target_proxy)]
        self.publisher_.publish(msg)

    def heuristic_prediction(self, feat_dict):
        if feat_dict['failure_detected_last'] > 0.5:
            return 0.0
        if feat_dict['tracked_feature_count_mgr'] < 20:
            return 0.25
        if feat_dict['outlier_ratio_last'] > 0.20:
            return 0.20
        if feat_dict['outlier_ratio_last'] > 0.10:
            return 0.50
        return 0.95

    def feature_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 9:
            self.get_logger().warn('feature msg length < 9, ignore')
            return

        stamp_sec = float(msg.data[0])
        if stamp_sec <= self.last_stamp:
            return
        self.last_stamp = stamp_sec

        feat_dict = {
            'tracked_feature_count_raw': float(msg.data[1]),
            'tracked_feature_count_mgr': float(msg.data[2]),
            'current_is_keyframe': float(msg.data[3]),
            'outlier_count_last': float(msg.data[4]),
            'inlier_count_last': float(msg.data[5]),
            'outlier_ratio_last': float(msg.data[6]),
            'solver_time_ms_last': float(msg.data[7]),
            'failure_detected_last': float(msg.data[8]),
        }

        if self.use_dummy_prediction:
            w_pred = self.heuristic_prediction(feat_dict)
            self.publish_prediction(stamp_sec, w_pred, -1.0)
            return

        x = []
        for name in RUNTIME_FEATURE_NAMES:
            val = feat_dict.get(name, self.fill_values.get(name, 0.0))
            if not np.isfinite(val):
                val = self.fill_values.get(name, 0.0)
            x.append(float(val))
        x = np.asarray(x, dtype=np.float32)

        self.feature_buffer.append(x)

        if len(self.feature_buffer) < self.seq_len:
            # warmup: 資料還不夠，先全部放行
            self.publish_prediction(stamp_sec, 1.0, -1.0)
            return

        x_seq = np.stack(list(self.feature_buffer), axis=0).astype(np.float32)
        x_seq = (x_seq - self.standardize_mean[None, :]) / self.standardize_std[None, :]
        x_seq = np.expand_dims(x_seq, axis=0)

        with torch.no_grad():
            xb = torch.from_numpy(x_seq).float().to(self.device)
            pred = self.model(xb).detach().cpu().numpy().reshape(-1)[0]
        pred = float(np.clip(pred, 0.0, 1.0))

        self.publish_prediction(stamp_sec, pred, -1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ReliabilityInferNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
