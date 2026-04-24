import argparse
import json
from pathlib import Path

from keyframe_selector import KeyframeSelectionBatchRunner
from extract_features import ReliabilityFeatureExtractor
from build_reliability_labels import ReliabilityLabelBuilder
from build_reliability_dataset import ReliabilityDatasetBuilder
from train_reliability_model import ReliabilityModelTrainer
from infer_reliability import ReliabilityInferencer


class ReliabilityPipelineRunner:
    """
    v2 pipeline:
      0. keyframe_selector.py -> keyframes/all_candidate_pairs.csv
      1. extract_features.py -> features/all_candidate_features.csv
      2. build_reliability_labels.py -> reliability_labels/reliability_labels.csv
      3. build_reliability_dataset.py -> reliability_dataset/{train,val,test}.npz
      4. train_reliability_model.py -> model_runs/<model_type>/best.pt
      5. infer_reliability.py -> reliability_inference/reliability_predictions.csv
    """

    def __init__(
        self,
        sequence_dir,

        # ===== 共用路徑 =====
        frames_csv='',
        images_dir='',
        candidate_csv_path='',
        feature_csv='',
        label_csv='',
        dataset_dir='',
        checkpoint='',
        inference_out_dir='',
        camera_info='',

        # ===== stage 開關 =====
        run_keyframe_selection=True,
        run_feature_extraction=True,
        run_label_build=True,
        run_dataset_build=True,
        run_training=True,
        run_inference=True,

        # ===== keyframe_selector.py =====
        prefilter_translation_m=0.20,
        prefilter_rotation_deg=2.0,
        min_matches=120,
        min_inlier_ratio=0.30,
        min_translation_m=0.50,
        min_rotation_deg=5.0,
        grid_rows=4,
        grid_cols=4,
        min_matches_for_geometry=8,

        # ===== extract_features.py =====
        feature_out_dir='',
        feature_out_csv_name='all_candidate_features.csv',
        clean_feature_output_dir=False,

        # ===== build_reliability_labels.py =====
        keyframes_csv='',
        label_out_dir='',
        label_mode='hybrid',
        alpha_hybrid=0.50,
        helpful_thr=0.60,
        harmful_thr=0.40,
        rot_noise_deg=0.80,
        trans_noise_m=0.03,

        # ===== build_reliability_dataset.py =====
        seq_len=8,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_rows=32,
        purge_gap=None,
        use_validity_mask=True,
        require_label_cls=False,

        # ===== train_reliability_model.py =====
        trainer_out_dir='',
        model_type='gru',
        epochs=80,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=64,
        num_layers=1,
        dropout=0.10,
        seed=42,
        device=None,
        lambda_reg=1.0,
        lambda_cls=1.0,
        label_smoothing=0.0,
        grad_clip_norm=1.0,
        patience=12,
        min_delta=1e-5,
        num_workers=0,

        # ===== infer_reliability.py =====
        tau=0.5,
        helpful_prob_thr=0.5,
        infer_batch_size=256,
    ):
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()

        # shared paths
        self.frames_csv = Path(frames_csv).expanduser().resolve() if frames_csv else self.sequence_dir / 'keyframes' / 'keyframes.csv'
        self.images_dir = Path(images_dir).expanduser().resolve() if images_dir else self.sequence_dir / 'keyframes' / 'images'
        self.camera_info = Path(camera_info).expanduser().resolve() if camera_info else self.sequence_dir / 'camera_info.yaml'

        self.candidate_csv_path = (
            Path(candidate_csv_path).expanduser().resolve()
            if candidate_csv_path
            else self.sequence_dir / 'keyframes' / 'all_candidate_pairs.csv'
        )

        self.feature_out_dir = Path(feature_out_dir).expanduser().resolve() if feature_out_dir else self.sequence_dir / 'features'
        self.feature_out_csv_name = str(feature_out_csv_name)
        self.feature_csv = (
            Path(feature_csv).expanduser().resolve()
            if feature_csv
            else self.feature_out_dir / self.feature_out_csv_name
        )
        self.clean_feature_output_dir = bool(clean_feature_output_dir)

        self.keyframes_csv = (
            Path(keyframes_csv).expanduser().resolve()
            if keyframes_csv
            else self.sequence_dir / 'keyframes' / 'keyframes.csv'
        )

        self.label_out_dir = Path(label_out_dir).expanduser().resolve() if label_out_dir else self.sequence_dir / 'reliability_labels'
        self.label_csv = (
            Path(label_csv).expanduser().resolve()
            if label_csv
            else self.label_out_dir / 'reliability_labels.csv'
        )

        self.dataset_dir = Path(dataset_dir).expanduser().resolve() if dataset_dir else self.sequence_dir / 'reliability_dataset'
        self.model_type = str(model_type)
        self.trainer_out_dir = (
            Path(trainer_out_dir).expanduser().resolve()
            if trainer_out_dir
            else self.dataset_dir / 'model_runs' / self.model_type
        )
        self.checkpoint = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint
            else self.trainer_out_dir / 'best.pt'
        )
        self.inference_out_dir = (
            Path(inference_out_dir).expanduser().resolve()
            if inference_out_dir
            else self.sequence_dir / 'reliability_inference'
        )

        # stage switches
        self.run_keyframe_selection = bool(run_keyframe_selection)
        self.run_feature_extraction = bool(run_feature_extraction)
        self.run_label_build = bool(run_label_build)
        self.run_dataset_build = bool(run_dataset_build)
        self.run_training = bool(run_training)
        self.run_inference = bool(run_inference)

        # selector params
        self.prefilter_translation_m = float(prefilter_translation_m)
        self.prefilter_rotation_deg = float(prefilter_rotation_deg)
        self.min_matches = int(min_matches)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_translation_m = float(min_translation_m)
        self.min_rotation_deg = float(min_rotation_deg)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.min_matches_for_geometry = int(min_matches_for_geometry)

        # label params
        self.label_mode = str(label_mode)
        self.alpha_hybrid = float(alpha_hybrid)
        self.helpful_thr = float(helpful_thr)
        self.harmful_thr = float(harmful_thr)
        self.rot_noise_deg = float(rot_noise_deg)
        self.trans_noise_m = float(trans_noise_m)

        # dataset params
        self.seq_len = int(seq_len)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.min_rows = int(min_rows)
        self.purge_gap = None if purge_gap is None else int(purge_gap)
        self.use_validity_mask = bool(use_validity_mask)
        self.require_label_cls = bool(require_label_cls)

        # trainer params
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.device = device
        self.lambda_reg = float(lambda_reg)
        self.lambda_cls = float(lambda_cls)
        self.label_smoothing = float(label_smoothing)
        self.grad_clip_norm = float(grad_clip_norm)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.num_workers = int(num_workers)

        # infer params
        self.tau = float(tau)
        self.helpful_prob_thr = float(helpful_prob_thr)
        self.infer_batch_size = int(infer_batch_size)

    def run_keyframe_stage(self):
        print('========== Stage 0/6: Build Candidate Pairs ==========')
        runner = KeyframeSelectionBatchRunner(
            sequence_dir=str(self.sequence_dir),
            frames_csv=str(self.frames_csv),
            images_dir=str(self.images_dir),
            out_csv=str(self.candidate_csv_path),
            camera_info=str(self.camera_info),
            prefilter_translation_m=self.prefilter_translation_m,
            prefilter_rotation_deg=self.prefilter_rotation_deg,
            min_matches=self.min_matches,
            min_inlier_ratio=self.min_inlier_ratio,
            min_translation_m=self.min_translation_m,
            min_rotation_deg=self.min_rotation_deg,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            min_matches_for_geometry=self.min_matches_for_geometry,
        )
        result = runner.run()
        self.candidate_csv_path = Path(result['out_csv']).expanduser().resolve()
        return result

    def run_feature_stage(self):
        print('========== Stage 1/6: Extract Features ==========')
        extractor = ReliabilityFeatureExtractor(
            sequence_dir=str(self.sequence_dir),
            candidate_csv_path=str(self.candidate_csv_path),
            out_dir=str(self.feature_out_dir),
            out_csv_name=self.feature_out_csv_name,
            clean_output_dir=self.clean_feature_output_dir,
        )
        result = extractor.run()
        self.feature_csv = Path(result['feature_csv']).expanduser().resolve()
        return result

    def run_label_stage(self):
        print('========== Stage 2/6: Build Reliability Labels ==========')
        builder = ReliabilityLabelBuilder(
            sequence_dir=str(self.sequence_dir),
            feature_csv=str(self.feature_csv),
            keyframes_csv=str(self.keyframes_csv),
            camera_info=str(self.camera_info),
            out_dir=str(self.label_out_dir),
            label_mode=self.label_mode,
            alpha_hybrid=self.alpha_hybrid,
            helpful_thr=self.helpful_thr,
            harmful_thr=self.harmful_thr,
            rot_noise_deg=self.rot_noise_deg,
            trans_noise_m=self.trans_noise_m,
            min_matches_for_geometry=self.min_matches_for_geometry,
            seed=self.seed,
        )
        result = builder.build()
        self.label_csv = Path(result['label_csv']).expanduser().resolve()
        return result

    def run_dataset_stage(self):
        print('========== Stage 3/6: Build Reliability Dataset ==========')
        builder = ReliabilityDatasetBuilder(
            sequence_dir=str(self.sequence_dir),
            feature_csv=str(self.feature_csv),
            label_csv=str(self.label_csv),
            out_dir=str(self.dataset_dir),
            seq_len=self.seq_len,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            min_rows=self.min_rows,
            purge_gap=self.purge_gap,
            use_validity_mask=self.use_validity_mask,
            require_label_cls=self.require_label_cls,
        )
        result = builder.build()
        self.dataset_dir = Path(result['out_dir']).expanduser().resolve()
        return result

    def run_training_stage(self):
        print('========== Stage 4/6: Train Reliability Model ==========')
        trainer = ReliabilityModelTrainer(
            dataset_dir=str(self.dataset_dir),
            out_dir=str(self.trainer_out_dir),
            model_type=self.model_type,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seed=self.seed,
            device=self.device,
            lambda_reg=self.lambda_reg,
            lambda_cls=self.lambda_cls,
            label_smoothing=self.label_smoothing,
            grad_clip_norm=self.grad_clip_norm,
            patience=self.patience,
            min_delta=self.min_delta,
            num_workers=self.num_workers,
        )
        result = trainer.run()
        if 'best_path' in result:
            self.checkpoint = Path(result['best_path']).expanduser().resolve()
        return result

    def run_inference_stage(self):
        print('========== Stage 5/6: Infer Reliability ==========')
        inferencer = ReliabilityInferencer(
            sequence_dir=str(self.sequence_dir),
            feature_csv=str(self.feature_csv),
            dataset_dir=str(self.dataset_dir),
            checkpoint=str(self.checkpoint),
            out_dir=str(self.inference_out_dir),
            label_csv=str(self.label_csv),
            tau=self.tau,
            helpful_prob_thr=self.helpful_prob_thr,
            batch_size=self.infer_batch_size,
            device=self.device,
            model_type=self.model_type,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=3,
        )
        result = inferencer.run()
        return result

    def run(self):
        summary = {
            'sequence_dir': str(self.sequence_dir),
            'stage_results': {},
        }

        if self.run_keyframe_selection:
            summary['stage_results']['keyframe_selection'] = self.run_keyframe_stage()
        else:
            print('========== Stage 0/6: Build Candidate Pairs (skip) ==========')

        if self.run_feature_extraction:
            summary['stage_results']['feature_extraction'] = self.run_feature_stage()
        else:
            print('========== Stage 1/6: Extract Features (skip) ==========')

        if self.run_label_build:
            summary['stage_results']['label_build'] = self.run_label_stage()
        else:
            print('========== Stage 2/6: Build Reliability Labels (skip) ==========')

        if self.run_dataset_build:
            summary['stage_results']['dataset_build'] = self.run_dataset_stage()
        else:
            print('========== Stage 3/6: Build Reliability Dataset (skip) ==========')

        if self.run_training:
            summary['stage_results']['training'] = self.run_training_stage()
        else:
            print('========== Stage 4/6: Train Reliability Model (skip) ==========')

        if self.run_inference:
            summary['stage_results']['inference'] = self.run_inference_stage()
        else:
            print('========== Stage 5/6: Infer Reliability (skip) ==========')

        summary['final_paths'] = {
            'frames_csv': str(self.frames_csv),
            'images_dir': str(self.images_dir),
            'candidate_csv_path': str(self.candidate_csv_path),
            'feature_csv': str(self.feature_csv),
            'label_csv': str(self.label_csv),
            'dataset_dir': str(self.dataset_dir),
            'checkpoint': str(self.checkpoint),
            'inference_out_dir': str(self.inference_out_dir),
        }

        summary_path = self.sequence_dir / 'pipeline_summary_v2.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print('========== Pipeline Finished ==========')
        print(f'pipeline summary: {summary_path}')
        for stage_name, stage_result in summary['stage_results'].items():
            print(f'[{stage_name}]')
            if isinstance(stage_result, dict):
                for k, v in stage_result.items():
                    print(f'  {k}: {v}')

        return summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sequence_dir', type=str, required=True, help='例如: /mnt/sata4t/dataset/sequence_001')

    parser.add_argument('--frames_csv', type=str, default='')
    parser.add_argument('--images_dir', type=str, default='')
    parser.add_argument('--candidate_csv_path', type=str, default='')
    parser.add_argument('--feature_csv', type=str, default='')
    parser.add_argument('--label_csv', type=str, default='')
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--inference_out_dir', type=str, default='')
    parser.add_argument('--camera_info', type=str, default='')

    parser.add_argument('--skip_keyframe_selection', action='store_true')
    parser.add_argument('--skip_feature_extraction', action='store_true')
    parser.add_argument('--skip_label_build', action='store_true')
    parser.add_argument('--skip_dataset_build', action='store_true')
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_inference', action='store_true')

    parser.add_argument('--prefilter_translation_m', type=float, default=0.20)
    parser.add_argument('--prefilter_rotation_deg', type=float, default=2.0)
    parser.add_argument('--min_matches', type=int, default=120)
    parser.add_argument('--min_inlier_ratio', type=float, default=0.30)
    parser.add_argument('--min_translation_m', type=float, default=0.50)
    parser.add_argument('--min_rotation_deg', type=float, default=5.0)
    parser.add_argument('--grid_rows', type=int, default=4)
    parser.add_argument('--grid_cols', type=int, default=4)
    parser.add_argument('--min_matches_for_geometry', type=int, default=8)

    parser.add_argument('--feature_out_dir', type=str, default='')
    parser.add_argument('--feature_out_csv_name', type=str, default='all_candidate_features.csv')
    parser.add_argument('--clean_feature_output_dir', action='store_true')

    parser.add_argument('--keyframes_csv', type=str, default='')
    parser.add_argument('--label_out_dir', type=str, default='')
    parser.add_argument('--label_mode', type=str, default='hybrid', choices=['weak', 'hybrid'])
    parser.add_argument('--alpha_hybrid', type=float, default=0.50)
    parser.add_argument('--helpful_thr', type=float, default=0.60)
    parser.add_argument('--harmful_thr', type=float, default=0.40)
    parser.add_argument('--rot_noise_deg', type=float, default=0.80)
    parser.add_argument('--trans_noise_m', type=float, default=0.03)

    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_rows', type=int, default=32)
    parser.add_argument('--purge_gap', type=int, default=-1)
    parser.add_argument('--disable_validity_mask', action='store_true')
    parser.add_argument('--require_label_cls', action='store_true')

    parser.add_argument('--trainer_out_dir', type=str, default='')
    parser.add_argument('--model_type', type=str, default='gru', choices=['mlp', 'gru', 'tcn'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--helpful_prob_thr', type=float, default=0.5)
    parser.add_argument('--infer_batch_size', type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()

    purge_gap = None if args.purge_gap < 0 else int(args.purge_gap)
    device = args.device.strip() if args.device.strip() != '' else None

    runner = ReliabilityPipelineRunner(
        sequence_dir=args.sequence_dir,

        frames_csv=args.frames_csv,
        images_dir=args.images_dir,
        candidate_csv_path=args.candidate_csv_path,
        feature_csv=args.feature_csv,
        label_csv=args.label_csv,
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        inference_out_dir=args.inference_out_dir,
        camera_info=args.camera_info,

        run_keyframe_selection=not args.skip_keyframe_selection,
        run_feature_extraction=not args.skip_feature_extraction,
        run_label_build=not args.skip_label_build,
        run_dataset_build=not args.skip_dataset_build,
        run_training=not args.skip_training,
        run_inference=not args.skip_inference,

        prefilter_translation_m=args.prefilter_translation_m,
        prefilter_rotation_deg=args.prefilter_rotation_deg,
        min_matches=args.min_matches,
        min_inlier_ratio=args.min_inlier_ratio,
        min_translation_m=args.min_translation_m,
        min_rotation_deg=args.min_rotation_deg,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        min_matches_for_geometry=args.min_matches_for_geometry,

        feature_out_dir=args.feature_out_dir,
        feature_out_csv_name=args.feature_out_csv_name,
        clean_feature_output_dir=args.clean_feature_output_dir,

        keyframes_csv=args.keyframes_csv,
        label_out_dir=args.label_out_dir,
        label_mode=args.label_mode,
        alpha_hybrid=args.alpha_hybrid,
        helpful_thr=args.helpful_thr,
        harmful_thr=args.harmful_thr,
        rot_noise_deg=args.rot_noise_deg,
        trans_noise_m=args.trans_noise_m,

        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_rows=args.min_rows,
        purge_gap=purge_gap,
        use_validity_mask=not args.disable_validity_mask,
        require_label_cls=args.require_label_cls,

        trainer_out_dir=args.trainer_out_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        device=device,
        lambda_reg=args.lambda_reg,
        lambda_cls=args.lambda_cls,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,

        tau=args.tau,
        helpful_prob_thr=args.helpful_prob_thr,
        infer_batch_size=args.infer_batch_size,
    )
    runner.run()


if __name__ == '__main__':
    main()