#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np


def safe_corr(df, a, b):
    x = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return np.nan
    if x[a].std() < 1e-12 or x[b].std() < 1e-12:
        return np.nan
    return x.corr().iloc[0, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to reliability_predictions.csv")
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    df_all = pd.read_csv(args.csv)
    df_pred = df_all[df_all["has_prediction"] == 1].copy()

    # 有 prediction 但不一定有 GT label
    df_valid_reg = df_pred[df_pred["label_reg_gt"].notna()].copy()
    df_valid_fail = df_pred[
        df_pred["label_reg_gt"].notna() &
        (df_pred["y_fail_gt"] >= 0)
    ].copy()

    print("========== File ==========")
    print(args.csv)

    print("\n========== Prediction Count ==========")
    print("total rows           :", len(df_all))
    print("has prediction rows  :", len(df_pred))
    print("valid reg label rows :", len(df_valid_reg))
    print("valid fail rows      :", len(df_valid_fail))

    print("\n========== w_pred Distribution ==========")
    print(df_pred["w_pred"].describe())

    print("\n========== p_fail Distribution ==========")
    print(df_pred["p_fail"].describe())

    print("\n========== Corrected Correlation ==========")
    print("corr(w_pred, label_reg_gt)  :", safe_corr(df_valid_reg, "w_pred", "label_reg_gt"))
    print("corr(w_pred, drift_trans)   :", safe_corr(df_valid_reg, "w_pred", "future_drift_trans_m_gt"))
    print("corr(w_pred, drift_rot)     :", safe_corr(df_valid_reg, "w_pred", "future_drift_rot_deg_gt"))
    print("corr(p_fail, y_fail_gt)     :", safe_corr(df_valid_fail, "p_fail", "y_fail_gt"))

    print("\n========== w_pred Quantile Bins, valid labels only ==========")
    df_valid_reg["w_bin"] = pd.qcut(df_valid_reg["w_pred"], args.bins, duplicates="drop")
    bin_table = df_valid_reg.groupby("w_bin", observed=True)[[
        "w_pred",
        "label_reg_gt",
        "p_fail",
        "future_drift_trans_m_gt",
        "future_drift_rot_deg_gt",
    ]].mean()

    # y_fail_gt 要另外只用 valid fail 算
    fail_bin = df_valid_fail.copy()
    fail_bin["w_bin"] = pd.qcut(fail_bin["w_pred"], args.bins, duplicates="drop")
    fail_table = fail_bin.groupby("w_bin", observed=True)[["y_fail_gt"]].mean()

    print(bin_table.join(fail_table, how="left"))

    print("\n========== Mode Counts ==========")
    print(df_pred["mode_name"].value_counts(dropna=False))

    print("\n========== Predicted Class Counts ==========")
    print(df_pred["pred_class"].value_counts(dropna=False))

    print("\n========== Lowest w_pred with valid labels ==========")
    cols = [
        "timestamp",
        "w_pred",
        "p_fail",
        "pred_class",
        "visual_weight",
        "label_reg_gt",
        "y_fail_gt",
        "future_drift_trans_m_gt",
        "future_drift_rot_deg_gt",
    ]
    print(df_valid_reg.sort_values("w_pred")[cols].head(20).to_string(index=False))

    print("\n========== Highest w_pred with valid labels ==========")
    print(df_valid_reg.sort_values("w_pred", ascending=False)[cols].head(20).to_string(index=False))

    print("\n========== Threshold Check for Soft Weighting ==========")
    for thr in [0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]:
        low = df_valid_reg[df_valid_reg["w_pred"] < thr]
        high = df_valid_reg[df_valid_reg["w_pred"] >= thr]
        if len(low) == 0 or len(high) == 0:
            continue
        print(f"\n[w_pred < {thr:.2f}] count={len(low)}")
        print("  mean label_reg :", low["label_reg_gt"].mean())
        print("  mean drift_m   :", low["future_drift_trans_m_gt"].mean())
        print("  mean drift_deg :", low["future_drift_rot_deg_gt"].mean())

        print(f"[w_pred >= {thr:.2f}] count={len(high)}")
        print("  mean label_reg :", high["label_reg_gt"].mean())
        print("  mean drift_m   :", high["future_drift_trans_m_gt"].mean())
        print("  mean drift_deg :", high["future_drift_rot_deg_gt"].mean())


if __name__ == "__main__":
    main()
