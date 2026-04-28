#!/usr/bin/env python3
import csv
import math
import os
from pathlib import Path

import numpy as np

KAIST_EXTRACTED_ROOT = Path("/mnt/sata4t/datasets/kaist_complex_urban/extracted")
VINS_EVAL_ROOT = Path.home() / "vins_ws" / "eval" / "kaist"

def rotmat_to_quat(R: np.ndarray):
    q = np.empty(4, dtype=float)  # qx, qy, qz, qw
    trace = np.trace(R)

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    return q

def is_header(row):
    if not row:
        return False
    try:
        float(row[0].strip())
        return False
    except ValueError:
        return True

def convert_global_pose_csv_to_tum(input_csv: Path, output_tum: Path):
    line_count = 0

    with input_csv.open("r", newline="") as f_in, output_tum.open("w") as f_out:
        reader = csv.reader(f_in)

        for row in reader:
            if not row:
                continue
            if is_header(row):
                continue

            vals = [float(x.strip()) for x in row]

            if len(vals) != 13:
                raise ValueError(
                    f"{input_csv}: unexpected column count = {len(vals)}, expected 13"
                )

            ts = vals[0]
            if ts > 1e12:
                ts = ts * 1e-9  # ns -> s

            T = np.array([
                [vals[1], vals[2], vals[3], vals[4]],
                [vals[5], vals[6], vals[7], vals[8]],
                [vals[9], vals[10], vals[11], vals[12]],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=float)

            R = T[:3, :3]
            t = T[:3, 3]
            qx, qy, qz, qw = rotmat_to_quat(R)

            f_out.write(
                f"{ts:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
            line_count += 1

    return line_count

def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

def main():
    if not KAIST_EXTRACTED_ROOT.exists():
        raise FileNotFoundError(f"Not found: {KAIST_EXTRACTED_ROOT}")

    VINS_EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([p for p in KAIST_EXTRACTED_ROOT.iterdir() if p.is_dir()])
    if not seq_dirs:
        print("No sequence directories found.")
        return

    print(f"Found {len(seq_dirs)} extracted sequence directories.\n")

    converted = []
    skipped = []

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        pose_csv = seq_dir / "pose" / seq_name / "global_pose.csv"
        pose_tum = seq_dir / "pose" / seq_name / "global_pose.tum"

        if not pose_csv.exists():
            skipped.append((seq_name, "missing global_pose.csv"))
            continue

        try:
            num_lines = convert_global_pose_csv_to_tum(pose_csv, pose_tum)
        except Exception as e:
            skipped.append((seq_name, f"convert failed: {e}"))
            continue

        eval_seq_dir = VINS_EVAL_ROOT / seq_name
        eval_seq_dir.mkdir(parents=True, exist_ok=True)

        safe_symlink(pose_tum, eval_seq_dir / "global_pose.tum")
        safe_symlink(pose_csv, eval_seq_dir / "global_pose.csv")

        converted.append((seq_name, num_lines, str(pose_tum), str(eval_seq_dir / "global_pose.tum")))

    print("=== Converted sequences ===")
    for seq_name, num_lines, pose_tum_path, link_path in converted:
        print(f"[OK] {seq_name}")
        print(f"     lines      : {num_lines}")
        print(f"     tum saved  : {pose_tum_path}")
        print(f"     vins link  : {link_path}")

    print("\n=== Skipped sequences ===")
    if skipped:
        for seq_name, reason in skipped:
            print(f"[SKIP] {seq_name} -> {reason}")
    else:
        print("None")

if __name__ == "__main__":
    main()

