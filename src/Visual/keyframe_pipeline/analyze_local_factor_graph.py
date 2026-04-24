import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sequence_dir',
        type=str,
        required=True,
        help='例如: /mnt/sata4t/dataset/sequence_001'
    )
    parser.add_argument(
        '--backend_dir',
        type=str,
        default='',
        help='若留空，預設使用 sequence_dir/local_factor_graph'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='若留空，預設輸出到 backend_dir/analysis'
    )
    return parser.parse_args()


def read_csv_rows(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_mean(values, default=0.0):
    if len(values) == 0:
        return default
    return sum(values) / len(values)


def analyze_summary(summary_json: Path):
    with open(summary_json, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    results = summary['results']
    mode_rows = []

    for mode, info in results.items():
        metrics = info['metrics']
        mode_rows.append({
            'mode': mode,
            'position_rmse_m': float(metrics['position_rmse_m']),
            'position_mean_m': float(metrics['position_mean_m']),
            'position_median_m': float(metrics['position_median_m']),
            'position_final_m': float(metrics['position_final_m']),
            'rotation_mean_deg': float(metrics['rotation_mean_deg']),
            'rotation_median_deg': float(metrics['rotation_median_deg']),
            'rotation_final_deg': float(metrics['rotation_final_deg']),
            'num_iters': int(info.get('num_iters', 0)),
            'trajectory_csv': info.get('trajectory_csv', ''),
        })

    return summary, mode_rows


def analyze_factor_debug(factor_debug_csv: Path):
    rows = read_csv_rows(factor_debug_csv)

    by_mode = {}
    for row in rows:
        mode = row['mode']
        by_mode.setdefault(mode, []).append(row)

    factor_stats = {}
    for mode, items in by_mode.items():
        pose_ok_list = [to_int(r.get('pose_ok', 0), 0) for r in items]
        gate_pass_list = [to_int(r.get('gate_pass', 0), 0) for r in items]
        alpha_list = [to_float(r.get('alpha', 0.0), 0.0) for r in items]
        w_list = [to_float(r.get('w_pred', 0.0), 0.0) for r in items]
        inlier_list = [to_float(r.get('inlier_ratio', 0.0), 0.0) for r in items]
        match_list = [to_int(r.get('num_matches', 0), 0) for r in items]

        factor_stats[mode] = {
            'num_pairs': len(items),
            'pose_ok_ratio': safe_mean(pose_ok_list),
            'gate_pass_ratio': safe_mean(gate_pass_list),
            'alpha_mean': safe_mean(alpha_list),
            'alpha_nonzero_ratio': safe_mean([1 if a > 0 else 0 for a in alpha_list]),
            'w_pred_mean': safe_mean(w_list),
            'inlier_ratio_mean': safe_mean(inlier_list),
            'num_matches_mean': safe_mean(match_list),
        }

    return factor_stats


def analyze_optimization_history(opt_csv: Path):
    rows = read_csv_rows(opt_csv)

    by_mode = {}
    for row in rows:
        mode = row['mode']
        by_mode.setdefault(mode, []).append(row)

    opt_stats = {}
    for mode, items in by_mode.items():
        items_sorted = sorted(items, key=lambda r: to_int(r.get('iter', 0), 0))

        if len(items_sorted) == 0:
            opt_stats[mode] = {
                'num_iters': 0,
                'first_cost': 0.0,
                'last_cost': 0.0,
                'best_cost': 0.0,
                'cost_drop': 0.0,
            }
            continue

        costs = [to_float(r.get('cost', 0.0), 0.0) for r in items_sorted]
        opt_stats[mode] = {
            'num_iters': len(items_sorted),
            'first_cost': costs[0],
            'last_cost': costs[-1],
            'best_cost': min(costs),
            'cost_drop': costs[0] - min(costs),
        }

    return opt_stats


def choose_best_mode(mode_rows):
    # 第一優先 position RMSE，第二優先 final position，第三優先 rotation mean
    sorted_rows = sorted(
        mode_rows,
        key=lambda r: (
            r['position_rmse_m'],
            r['position_final_m'],
            r['rotation_mean_deg'],
        )
    )
    return sorted_rows[0], sorted_rows


def main():
    args = parse_args()

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    backend_dir = Path(args.backend_dir).expanduser().resolve() if args.backend_dir else sequence_dir / 'local_factor_graph'
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else backend_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = backend_dir / 'summary.json'
    factor_debug_csv = backend_dir / 'factor_debug.csv'
    optimization_csv = backend_dir / 'optimization_history.csv'

    if not summary_json.exists():
        raise FileNotFoundError(f'找不到 {summary_json}')
    if not factor_debug_csv.exists():
        raise FileNotFoundError(f'找不到 {factor_debug_csv}')
    if not optimization_csv.exists():
        raise FileNotFoundError(f'找不到 {optimization_csv}')

    summary, mode_rows = analyze_summary(summary_json)
    factor_stats = analyze_factor_debug(factor_debug_csv)
    opt_stats = analyze_optimization_history(optimization_csv)

    best_mode, sorted_rows = choose_best_mode(mode_rows)

    comparison_csv = out_dir / 'mode_comparison.csv'
    with open(comparison_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mode',
            'position_rmse_m',
            'position_mean_m',
            'position_median_m',
            'position_final_m',
            'rotation_mean_deg',
            'rotation_median_deg',
            'rotation_final_deg',
            'num_pairs',
            'pose_ok_ratio',
            'gate_pass_ratio',
            'alpha_mean',
            'alpha_nonzero_ratio',
            'w_pred_mean',
            'inlier_ratio_mean',
            'num_matches_mean',
            'opt_num_iters',
            'opt_first_cost',
            'opt_last_cost',
            'opt_best_cost',
            'opt_cost_drop',
        ])

        for row in sorted_rows:
            mode = row['mode']
            fstats = factor_stats.get(mode, {})
            ostats = opt_stats.get(mode, {})

            writer.writerow([
                mode,
                row['position_rmse_m'],
                row['position_mean_m'],
                row['position_median_m'],
                row['position_final_m'],
                row['rotation_mean_deg'],
                row['rotation_median_deg'],
                row['rotation_final_deg'],
                fstats.get('num_pairs', 0),
                fstats.get('pose_ok_ratio', 0.0),
                fstats.get('gate_pass_ratio', 0.0),
                fstats.get('alpha_mean', 0.0),
                fstats.get('alpha_nonzero_ratio', 0.0),
                fstats.get('w_pred_mean', 0.0),
                fstats.get('inlier_ratio_mean', 0.0),
                fstats.get('num_matches_mean', 0.0),
                ostats.get('num_iters', 0),
                ostats.get('first_cost', 0.0),
                ostats.get('last_cost', 0.0),
                ostats.get('best_cost', 0.0),
                ostats.get('cost_drop', 0.0),
            ])

    best_json = out_dir / 'best_mode.json'
    best_payload = {
        'best_mode': best_mode['mode'],
        'reason': {
            'position_rmse_m': best_mode['position_rmse_m'],
            'position_final_m': best_mode['position_final_m'],
            'rotation_mean_deg': best_mode['rotation_mean_deg'],
        },
        'all_modes_sorted': [row['mode'] for row in sorted_rows],
    }
    with open(best_json, 'w', encoding='utf-8') as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    report_json = out_dir / 'analysis_report.json'
    report = {
        'sequence_dir': str(sequence_dir),
        'backend_dir': str(backend_dir),
        'summary_source': str(summary_json),
        'factor_debug_source': str(factor_debug_csv),
        'optimization_history_source': str(optimization_csv),
        'best_mode': best_payload,
        'factor_stats': factor_stats,
        'optimization_stats': opt_stats,
        'notes': [
            '先看 position_rmse_m 與 position_final_m 判斷哪個模式最好。',
            '如果 visual_always 比 inertial_only 差，代表低品質 visual 真的會拖垮後端。',
            '如果 hard_gate 或 gate_and_weight 比 visual_always 好，代表 reliability estimator 有實際幫助。',
            '如果 soft_weight 或 gate_and_weight 最好，代表只做 reject 不夠，adaptive weighting 也有價值。',
        ]
    }
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'分析完成，輸出到: {out_dir}')
    print(f'mode comparison: {comparison_csv}')
    print(f'best mode: {best_json}')
    print(f'report: {report_json}')


if __name__ == '__main__':
    main()