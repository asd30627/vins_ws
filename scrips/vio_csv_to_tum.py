#!/usr/bin/env python3
import csv
import math
import os
import sys


def normalize_quat(qx, qy, qz, qw):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        return None
    return qx / norm, qy / norm, qz / norm, qw / norm


def convert_timestamp(ts_raw: float) -> float:
    """
    自動判斷 timestamp 單位：
    - 如果是像 1544590816602232320 這種，視為 nanoseconds，除以 1e9
    - 如果是像 1544590800.199765205 這種，視為 seconds，直接用
    """
    if ts_raw > 1e12:
        return ts_raw / 1e9
    return ts_raw


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 vio_csv_to_tum.py <input_vio.csv> <output_vio.tum>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_tum = sys.argv[2]

    if not os.path.isfile(input_csv):
        print(f"Input file not found: {input_csv}")
        sys.exit(1)

    rows_written = 0

    with open(input_csv, "r", newline="") as f_in, open(output_tum, "w") as f_out:
        reader = csv.reader(f_in)

        for row in reader:
            if not row:
                continue

            row = [x.strip() for x in row if x.strip() != ""]
            if len(row) < 8:
                continue

            try:
                ts_raw = float(row[0])
                ts = convert_timestamp(ts_raw)

                px = float(row[1])
                py = float(row[2])
                pz = float(row[3])

                # 依照 VINS 寫檔順序：qw, qx, qy, qz
                qw = float(row[4])
                qx = float(row[5])
                qy = float(row[6])
                qz = float(row[7])

                q = normalize_quat(qx, qy, qz, qw)
                if q is None:
                    continue
                qx, qy, qz, qw = q

                # TUM 格式：timestamp tx ty tz qx qy qz qw
                f_out.write(
                    f"{ts:.9f} {px:.9f} {py:.9f} {pz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
                )
                rows_written += 1

            except ValueError:
                continue

    print(f"Saved: {output_tum}")
    print(f"Rows written: {rows_written}")


if __name__ == "__main__":
    main()
