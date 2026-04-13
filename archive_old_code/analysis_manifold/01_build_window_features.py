# analysisV2/05_build_window_features.py  (PASTABLE FULL FILE)
import argparse
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def load_hr(hr_path: str) -> pd.DataFrame:
    df = pd.read_csv(hr_path)
    if "time" not in df.columns:
        raise ValueError(f"HR file missing 'time': {hr_path}")

    # Prefer clean HR if available
    if "hr_bpm_clean" in df.columns:
        hr_col = "hr_bpm_clean"
    elif "hr_bpm" in df.columns:
        hr_col = "hr_bpm"
    else:
        raise ValueError(f"HR file missing hr_bpm/hr_bpm_clean: {hr_path}")

    out = df[["time", hr_col]].copy()
    out.rename(columns={hr_col: "hr_bpm"}, inplace=True)

    # If is_valid exists, respect it (keep valid only)
    if "is_valid" in df.columns:
        is_valid = df["is_valid"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        out = out[is_valid].copy()

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "hr_bpm"])
    out["time"] = out["time"].astype(float)
    out["hr_bpm"] = out["hr_bpm"].astype(float)

    # Align to integer seconds
    out["t_sec"] = np.rint(out["time"].to_numpy()).astype(np.int64)
    out = out.groupby("t_sec", as_index=False)["hr_bpm"].median().sort_values("t_sec")
    return out


def load_hrv(hrv_path: str) -> pd.DataFrame:
    df = pd.read_csv(hrv_path)
    need = {"time", "sdnn_ms", "rmssd_ms"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"HRV file missing columns {need - set(df.columns)}: {hrv_path}")

    out_cols = ["time", "sdnn_ms", "rmssd_ms"]
    if "n_beats_in_window" in df.columns:
        out_cols.append("n_beats_in_window")

    out = df[out_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time"])
    out["time"] = out["time"].astype(float)
    out["t_sec"] = np.rint(out["time"].to_numpy()).astype(np.int64)

    # Collapse duplicates per second (median)
    agg = {
        "sdnn_ms": "median",
        "rmssd_ms": "median",
    }
    if "n_beats_in_window" in out.columns:
        agg["n_beats_in_window"] = "median"

    out = out.groupby("t_sec", as_index=False).agg(agg).sort_values("t_sec")
    return out


def window_stats(x: np.ndarray) -> Tuple[float, float, float]:
    """mean, std, p95-p05"""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    std = float(np.std(x))
    p95 = float(np.percentile(x, 95))
    p05 = float(np.percentile(x, 5))
    return (mean, std, p95 - p05)


def window_slope(t: np.ndarray, x: np.ndarray) -> float:
    """Slope of x vs t using simple linear fit. t in seconds."""
    ok = np.isfinite(t) & np.isfinite(x)
    if ok.sum() < 5:
        return np.nan
    tt = t[ok].astype(float)
    yy = x[ok].astype(float)
    tt = tt - np.mean(tt)
    try:
        coef = np.polyfit(tt, yy, 1)
        return float(coef[0])
    except Exception:
        return np.nan


def compute_window_features(merged: pd.DataFrame, window_s: int, step_s: int) -> pd.DataFrame:
    """
    merged must have columns:
      t_sec, hr_bpm, rmssd_ms, sdnn_ms, (optional n_beats_in_window)
    """
    if merged.empty:
        return pd.DataFrame()

    t0 = int(merged["t_sec"].min())
    t1 = int(merged["t_sec"].max())

    rows = []
    start = t0
    w = int(window_s)
    s = int(step_s)

    while start + w <= t1 + 1:
        end = start + w  # [start, end)
        wdf = merged[(merged["t_sec"] >= start) & (merged["t_sec"] < end)].copy()

        # Quality: require finite HR + finite HRV on same seconds
        finite_mask = np.isfinite(wdf["hr_bpm"].to_numpy()) & np.isfinite(wdf["rmssd_ms"].to_numpy()) & np.isfinite(wdf["sdnn_ms"].to_numpy())
        valid_pct = 100.0 * float(np.mean(finite_mask)) if len(wdf) > 0 else 0.0

        # Features (use only finite rows for HRV means)
        hr = wdf["hr_bpm"].to_numpy(dtype=float)
        t = wdf["t_sec"].to_numpy(dtype=float)

        hr_mean, hr_std, hr_p95_p05 = window_stats(hr)
        hr_slp = window_slope(t, hr)

        rm = wdf["rmssd_ms"].to_numpy(dtype=float)
        sd = wdf["sdnn_ms"].to_numpy(dtype=float)

        rm_mean, rm_std, _ = window_stats(rm)
        sd_mean, sd_std, _ = window_stats(sd)

        # Optional: beats count stats (median)
        beats_med = np.nan
        if "n_beats_in_window" in wdf.columns:
            bb = wdf["n_beats_in_window"].to_numpy(dtype=float)
            beats_med = float(np.nanmedian(bb)) if np.isfinite(bb).any() else np.nan

        rows.append({
            "t_start": float(start),
            "t_end": float(end),
            "t_center": float(start + w / 2.0),
            "n_samples": int(len(wdf)),
            "valid_pct": float(valid_pct),
            "hr_mean": hr_mean,
            "hr_std": hr_std,
            "hr_p95_p05": float(hr_p95_p05),
            "hr_slope": float(hr_slp),
            "rmssd_mean": float(rm_mean),
            "rmssd_std": float(rm_std),
            "sdnn_mean": float(sd_mean),
            "sdnn_std": float(sd_std),
            "beats_med": float(beats_med),
        })

        start += s

    return pd.DataFrame(rows)


def zscore_group(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = out[c].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            out[c] = x - mu
        else:
            out[c] = (x - mu) / sd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_root", default="analysis_outputs/hr_clean_v2")
    ap.add_argument("--hrv_root", default="analysis_outputs/hrv_v2")
    ap.add_argument("--out_root", default="analysis_outputs/manifold_v2")
    ap.add_argument("--segments", default="baseline_warmup,meditation_2A,meditation_2B")
    ap.add_argument("--window_s", type=int, default=60)
    ap.add_argument("--step_s", type=int, default=5)
    ap.add_argument("--min_valid_pct", type=float, default=80.0, help="Drop windows below this valid_pct (default 80)")
    args = ap.parse_args()

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = sorted([Path(p).name for p in glob.glob(os.path.join(args.hr_root, "EmotiBit_*")) if Path(p).is_dir()])
    if not subjects:
        raise SystemExit(f"No EmotiBit_* found under {args.hr_root}")

    all_rows = []
    missing = 0

    for subj in subjects:
        for seg in segments:
            hr_path = os.path.join(args.hr_root, subj, seg, "HR_1Hz_clean.csv")
            hrv_path = os.path.join(args.hrv_root, subj, seg, "HRV_1Hz.csv")
            if not (os.path.exists(hr_path) and os.path.exists(hrv_path)):
                missing += 1
                continue

            hr = load_hr(hr_path)
            hrv = load_hrv(hrv_path)

            # Merge on t_sec (exact 1Hz alignment after rounding)
            merged = pd.merge(hr, hrv, on="t_sec", how="inner")
            if merged.empty:
                continue

            feat = compute_window_features(merged, window_s=args.window_s, step_s=args.step_s)
            if feat.empty:
                continue

            # metadata
            feat["subject"] = subj
            feat["segment"] = seg

            # relative time per subject+segment (for trajectories)
            feat = feat.sort_values("t_center").reset_index(drop=True)
            feat["t_rel"] = feat["t_center"] - float(feat["t_center"].iloc[0])

            # drop low-quality windows
            feat = feat[feat["valid_pct"] >= float(args.min_valid_pct)].copy()
            if feat.empty:
                continue

            all_rows.append(feat)

    if not all_rows:
        raise SystemExit("No windows produced. Check paths/segments and min_valid_pct.")

    df = pd.concat(all_rows, ignore_index=True)

    # Feature columns we will embed
    feature_cols = [
        "hr_mean", "hr_std", "hr_p95_p05", "hr_slope",
        "rmssd_mean", "rmssd_std",
        "sdnn_mean", "sdnn_std",
        # beats_med is optional; keep it, but it can be noisier
        "beats_med",
    ]

    # Save raw feature table
    out_raw = out_root / "features_windows.csv"
    df.to_csv(out_raw, index=False)

    # Within-subject z-score (best for state/trajectory)
    df_within = df.copy()
    df_within[feature_cols] = np.nan  # placeholder
    parts = []
    for subj, g in df.groupby("subject", sort=False):
        gz = g.copy()
        gz = zscore_group(gz, feature_cols)
        parts.append(gz)
    df_within = pd.concat(parts, ignore_index=True)
    out_within = out_root / "features_windows_within_z.csv"
    df_within.to_csv(out_within, index=False)

    # Global z-score (keeps between-subject differences)
    df_global = zscore_group(df.copy(), feature_cols)
    out_global = out_root / "features_windows_global_z.csv"
    df_global.to_csv(out_global, index=False)

    print(f"Saved raw windows: {out_raw}")
    print(f"Saved within-subject z: {out_within}")
    print(f"Saved global z: {out_global}")

    # Quick sanity prints
    print("\nCounts:")
    print(df.groupby(["segment"])["subject"].nunique().rename("n_subjects"))
    print(df.groupby(["segment"]).size().rename("n_windows"))

    print("\nExample rows:")
    show = ["subject", "segment", "t_center", "t_rel", "valid_pct"] + feature_cols
    print(df[show].head(8).to_string(index=False))


if __name__ == "__main__":
    main()


"""
Run (recommended):
python3 analysis_manifold/01_build_window_features.py \
  --hr_root analysis_outputs/hr_clean_v2 \
  --hrv_root analysis_outputs/hrv_v2 \
  --out_root analysis_outputs/manifold_v2 \
  --segments baseline_warmup,meditation_2A,meditation_2B \
  --window_s 60 --step_s 5 --min_valid_pct 80
"""