#!/usr/bin/env python3
import csv
import math
import os
import itertools
from bisect import bisect_left
from decimal import Decimal, InvalidOperation
from statistics import mean


def parse_int_lossless(s: str):
    s = s.strip()
    if not s:
        return None
    if s.lstrip("+-").isdigit():
        try:
            return int(s)
        except Exception:
            return None
    try:
        return int(Decimal(s))
    except (InvalidOperation, ValueError):
        return None


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_xsens(csv_path):
    samples = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    start_idx = 0
    if rows and any(not is_number(c.strip()) for c in rows[0] if c.strip()):
        start_idx = 1

    for row in rows[start_idx:]:
        if not row or len(row) < 14:
            continue
        try:
            ts_ns = parse_int_lossless(row[0])
            if ts_ns is None:
                continue

            # 依你現在 kaist_player_node.py 的解析
            gx = float(row[8].strip())
            gy = float(row[9].strip())
            gz = float(row[10].strip())

            ax = float(row[11].strip())
            ay = float(row[12].strip())
            az = float(row[13].strip())

            samples.append((ts_ns, ax, ay, az, gx, gy, gz))
        except Exception:
            continue
    return samples


def load_fog(csv_path):
    samples = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                ts_ns = parse_int_lossless(row[0])
                if ts_ns is None:
                    continue
                gx = float(row[1].strip())
                gy = float(row[2].strip())
                gz = float(row[3].strip())
                samples.append((ts_ns, gx, gy, gz))
            except Exception:
                continue
    return samples


def nearest_match(xsens_samples, fog_samples):
    fog_ts = [x[0] for x in fog_samples]
    matched = []

    for xs in xsens_samples:
        ts = xs[0]
        idx = bisect_left(fog_ts, ts)
        cand = []
        if idx < len(fog_samples):
            cand.append(fog_samples[idx])
        if idx > 0:
            cand.append(fog_samples[idx - 1])
        if not cand:
            continue
        fg = min(cand, key=lambda s: abs(s[0] - ts))
        matched.append((xs, fg, abs(fg[0] - ts)))
    return matched


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def corr(a, b):
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    ma = mean(a)
    mb = mean(b)
    სა = 0.0
    sb = 0.0
    sc = 0.0
    for x, y in zip(a, b):
        dx = x - ma
        dy = y - mb
        sc += dx * dy
        სა += dx * dx
        sb += dy * dy
    if სა <= 1e-12 or sb <= 1e-12:
        return 0.0
    return sc / math.sqrt(სა * sb)


def rmse(a, b):
    if len(a) != len(b) or len(a) == 0:
        return float("inf")
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s / len(a))


def apply_mapping(v, perm, signs, scale):
    return [
        scale * signs[0] * v[perm[0]],
        scale * signs[1] * v[perm[1]],
        scale * signs[2] * v[perm[2]],
    ]


def evaluate_mapping(matched, perm, signs, scale, bias=None):
    xs_x, xs_y, xs_z = [], [], []
    fg_x, fg_y, fg_z = [], [], []

    for xs, fg, _ in matched:
        xs_g = [xs[4], xs[5], xs[6]]
        fg_g = [fg[1], fg[2], fg[3]]

        mapped = apply_mapping(fg_g, perm, signs, scale)
        if bias is not None:
            mapped = [mapped[i] - bias[i] for i in range(3)]

        xs_x.append(xs_g[0]); xs_y.append(xs_g[1]); xs_z.append(xs_g[2])
        fg_x.append(mapped[0]); fg_y.append(mapped[1]); fg_z.append(mapped[2])

    c1 = corr(xs_x, fg_x)
    c2 = corr(xs_y, fg_y)
    c3 = corr(xs_z, fg_z)

    r1 = rmse(xs_x, fg_x)
    r2 = rmse(xs_y, fg_y)
    r3 = rmse(xs_z, fg_z)

    score = (c1 + c2 + c3) - 0.1 * (r1 + r2 + r3)

    return {
        "score": score,
        "corr": (c1, c2, c3),
        "rmse": (r1, r2, r3),
    }


def estimate_bias(matched, perm, signs, scale, seconds_for_bias=5.0):
    if not matched:
        return [0.0, 0.0, 0.0]

    t0 = matched[0][0][0]
    buf = []
    for xs, fg, _ in matched:
        if (xs[0] - t0) / 1e9 > seconds_for_bias:
            break
        xs_g = [xs[4], xs[5], xs[6]]
        fg_g = [fg[1], fg[2], fg[3]]
        mapped = apply_mapping(fg_g, perm, signs, scale)
        buf.append([mapped[i] - xs_g[i] for i in range(3)])

    if not buf:
        return [0.0, 0.0, 0.0]

    bx = mean([x[0] for x in buf])
    by = mean([x[1] for x in buf])
    bz = mean([x[2] for x in buf])
    return [bx, by, bz]


def main():
    dataset_root = "/mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo"
    xsens_csv = os.path.join(dataset_root, "data/urban28-pankyo/sensor_data/xsens_imu.csv")
    fog_csv = os.path.join(dataset_root, "data/urban28-pankyo/sensor_data/fog.csv")

    print(f"xsens_csv = {xsens_csv}")
    print(f"fog_csv   = {fog_csv}")

    xsens = load_xsens(xsens_csv)
    fog = load_fog(fog_csv)

    print(f"xsens samples = {len(xsens)}")
    print(f"fog samples   = {len(fog)}")

    matched = nearest_match(xsens, fog)
    print(f"matched pairs = {len(matched)}")

    if not matched:
        print("No matched samples.")
        return

    dt_ms = [m[2] / 1e6 for m in matched]
    print(f"nearest dt avg = {mean(dt_ms):.6f} ms")
    print(f"nearest dt max = {max(dt_ms):.6f} ms")

    # 先粗看量級
    xs_norm = mean([vec_norm([x[4], x[5], x[6]]) for x, _, _ in matched[:5000]])
    fg_norm = mean([vec_norm([f[1], f[2], f[3]]) for _, f, _ in matched[:5000]])
    print(f"mean gyro norm xsens = {xs_norm:.8f}")
    print(f"mean gyro norm fog   = {fg_norm:.8f}")
    if fg_norm > 1e-12:
        print(f"norm ratio xsens/fog = {xs_norm / fg_norm:.8f}")

    scale_candidates = {
        "identity": 1.0,
        "deg_to_rad": math.pi / 180.0,
        "rad_to_deg": 180.0 / math.pi,
    }

    best = None

    for scale_name, scale in scale_candidates.items():
        for perm in itertools.permutations([0, 1, 2]):
            for signs in itertools.product([1, -1], repeat=3):
                res = evaluate_mapping(matched, perm, signs, scale, bias=None)
                item = {
                    "scale_name": scale_name,
                    "scale": scale,
                    "perm": perm,
                    "signs": signs,
                    "bias": None,
                    **res,
                }
                if best is None or item["score"] > best["score"]:
                    best = item

    print("\n=== Best mapping without bias ===")
    print(f"scale_name = {best['scale_name']}")
    print(f"scale      = {best['scale']}")
    print(f"perm       = {best['perm']}  # fog axis -> xsens axis")
    print(f"signs      = {best['signs']}")
    print(f"corr xyz   = {best['corr']}")
    print(f"rmse xyz   = {best['rmse']}")
    print(f"score      = {best['score']:.8f}")

    bias = estimate_bias(matched, best["perm"], best["signs"], best["scale"], seconds_for_bias=5.0)
    res_bias = evaluate_mapping(matched, best["perm"], best["signs"], best["scale"], bias=bias)

    print("\n=== Same mapping with initial 5s bias removed ===")
    print(f"bias xyz   = {bias}")
    print(f"corr xyz   = {res_bias['corr']}")
    print(f"rmse xyz   = {res_bias['rmse']}")
    print(f"score      = {res_bias['score']:.8f}")

    print("\n=== Recommendation ===")
    print("1. If best scale is deg_to_rad, fog gyro is likely in deg/s.")
    print("2. If perm != (0,1,2) or any sign is -1, fog frame is not same as xsens frame.")
    print("3. If bias is large, subtract bias before publishing.")
    print("4. If corr is still low after mapping/bias, do NOT directly use fog_xsens in VINS.")

    print("\n=== Example converted fog gyro formula ===")
    p = best["perm"]
    s = best["signs"]
    sc = best["scale"]
    print(
        f"mapped_gx = {sc:.12f} * ({s[0]}) * fog[{p[0]}] - ({bias[0]:.12f})\n"
        f"mapped_gy = {sc:.12f} * ({s[1]}) * fog[{p[1]}] - ({bias[1]:.12f})\n"
        f"mapped_gz = {sc:.12f} * ({s[2]}) * fog[{p[2]}] - ({bias[2]:.12f})"
    )


if __name__ == "__main__":
    main()
