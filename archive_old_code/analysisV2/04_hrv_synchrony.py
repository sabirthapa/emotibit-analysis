# analysisV2/04_hrv_synchrony.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Loading + alignment
# -----------------------------
def load_hrv_csv(path: Path, feature: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {path}")
    if feature not in df.columns:
        raise ValueError(f"Missing '{feature}' column in {path}. Found: {list(df.columns)}")

    out = df[["time", feature]].copy()
    out.rename(columns={feature: "x"}, inplace=True)

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "x"])
    out["time"] = out["time"].astype(float)
    out["x"] = out["x"].astype(float)
    return out


def align_to_seconds(df: pd.DataFrame) -> pd.Series:
    """
    Convert float timestamps to integer seconds using rounding.
    Collapse duplicates by taking median per second.
    """
    t_sec = np.rint(df["time"].to_numpy()).astype(np.int64)
    tmp = pd.DataFrame({"t_sec": t_sec, "x": df["x"].to_numpy()})
    s = tmp.groupby("t_sec")["x"].median().sort_index()
    return s


def find_emotibit_ids(root: Path) -> List[str]:
    return sorted([p.name for p in root.glob("EmotiBit_*") if p.is_dir()])


def segment_file(hrv_root: Path, emotibit: str, segment: str) -> Path:
    return hrv_root / emotibit / segment / "HRV_1Hz.csv"


def build_aligned_matrix(
    hrv_root: Path,
    segment: str,
    emotibits: List[str],
    feature: str,
    min_subjects: int,
) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}

    for eb in emotibits:
        f = segment_file(hrv_root, eb, segment)
        if not f.exists():
            continue

        df = load_hrv_csv(f, feature=feature)
        s = align_to_seconds(df)
        if len(s) >= 10:
            series_map[eb] = s

    if len(series_map) < 2:
        return pd.DataFrame()

    # Common overlap range
    starts = [int(s.index.min()) for s in series_map.values()]
    ends = [int(s.index.max()) for s in series_map.values()]
    t_start = max(starts)
    t_end = min(ends)
    if t_end - t_start < 10:
        return pd.DataFrame()

    idx = pd.Index(np.arange(t_start, t_end + 1, dtype=np.int64), name="t_sec")
    mat = pd.DataFrame({eb: series_map[eb].reindex(idx) for eb in sorted(series_map.keys())})

    # Keep seconds where at least min_subjects have data
    keep = mat.notna().sum(axis=1) >= int(min_subjects)
    mat = mat.loc[keep]

    return mat


# -----------------------------
# Metrics
# -----------------------------
def mean_pairwise_corr(mat: pd.DataFrame, min_periods: int = 10) -> float:
    if mat.shape[1] < 2:
        return float("nan")

    corr = mat.corr(method="pearson", min_periods=min_periods)
    vals = corr.to_numpy()
    triu = vals[np.triu_indices_from(vals, k=1)]
    triu = triu[np.isfinite(triu)]
    return float(np.mean(triu)) if len(triu) else float("nan")


def leave_one_out_isc(mat: pd.DataFrame, min_points: int = 10) -> float:
    """
    For each subject i:
      corr( subject_i , mean(other_subjects) )
    Then average across subjects.
    """
    if mat.shape[1] < 2:
        return float("nan")

    arr = mat.to_numpy(dtype=float)
    out = []
    for i in range(arr.shape[1]):
        x = arr[:, i]
        others = np.delete(arr, i, axis=1)
        y = np.nanmean(others, axis=1)

        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_points:
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isfinite(r):
            out.append(r)

    return float(np.mean(out)) if out else float("nan")


def max_lag_corr_pair(x: np.ndarray, y: np.ndarray, max_lag_s: int, min_points: int = 10) -> Tuple[float, int]:
    """
    Best correlation within +/- max_lag_s seconds.
    lag > 0 means y is shifted forward (happens later).
    """
    best_r = -np.inf
    best_lag = 0

    for lag in range(-max_lag_s, max_lag_s + 1):
        if lag < 0:
            xs = x[-lag:]
            ys = y[: len(xs)]
        elif lag > 0:
            ys = y[lag:]
            xs = x[: len(ys)]
        else:
            xs = x
            ys = y

        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < min_points:
            continue

        r = np.corrcoef(xs[ok], ys[ok])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_lag = lag

    return (float(best_r) if np.isfinite(best_r) else float("nan"), int(best_lag))


def bestlag_corr_matrix(mat: pd.DataFrame, max_lag_s: int, min_points: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      best_r_matrix (NxN) : best correlation per pair
      best_lag_matrix (NxN): lag (seconds) at which best_r occurs
    """
    cols = list(mat.columns)
    arr = mat.to_numpy(dtype=float)

    R = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    L = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, lag = max_lag_corr_pair(arr[:, i], arr[:, j], max_lag_s, min_points=min_points)
            R.iloc[i, j] = r
            R.iloc[j, i] = r
            L.iloc[i, j] = lag
            L.iloc[j, i] = -lag  # opposite direction for symmetry

    return R, L


def mean_pairwise_bestlag(mat: pd.DataFrame, max_lag_s: int, min_points: int = 10) -> Tuple[float, float]:
    """
    Mean of best correlation across pairs and mean abs lag.
    """
    if mat.shape[1] < 2:
        return float("nan"), float("nan")

    cols = list(mat.columns)
    arr = mat.to_numpy(dtype=float)

    best_rs = []
    best_lags = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, lag = max_lag_corr_pair(arr[:, i], arr[:, j], max_lag_s, min_points=min_points)
            if np.isfinite(r):
                best_rs.append(r)
                best_lags.append(abs(lag))

    return (float(np.mean(best_rs)) if best_rs else float("nan"),
            float(np.mean(best_lags)) if best_lags else float("nan"))


def sliding_window_synchrony(mat: pd.DataFrame, window_s: int, step_s: int, min_periods: int = 10) -> pd.DataFrame:
    if mat.empty:
        return pd.DataFrame(columns=["t_center", "mean_pairwise_r", "t_start", "t_end"])

    t = mat.index.to_numpy()
    t0, t1 = int(t.min()), int(t.max())

    rows = []
    w = int(window_s)
    s = int(step_s)

    for start in range(t0, t1 - w + 1, s):
        end = start + w
        seg = mat.loc[(mat.index >= start) & (mat.index < end)]
        r = mean_pairwise_corr(seg, min_periods=min_periods) if len(seg) >= min_periods else float("nan")
        rows.append({"t_center": start + w / 2.0, "mean_pairwise_r": r, "t_start": start, "t_end": end})

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------
def plot_heatmap(mat: pd.DataFrame, out_png: Path, title: str, vmin=None, vmax=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    plt.yticks(range(len(mat.index)), mat.index)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_windowed_sync(sync_df: pd.DataFrame, out_png: Path, title: str):
    plt.figure(figsize=(7, 3))
    if not sync_df.empty:
        plt.plot(sync_df["t_center"], sync_df["mean_pairwise_r"])
    plt.xlabel("time (s, aligned)")
    plt.ylabel("mean pairwise r")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------
# Segment processing
# -----------------------------
def process_segment(
    hrv_root: Path,
    out_root: Path,
    segment: str,
    emotibits: List[str],
    feature: str,
    window_s: int,
    step_s: int,
    max_lag_s: int,
    min_subjects: int,
    min_periods: int,
) -> Dict:
    seg_out = out_root / segment
    seg_out.mkdir(parents=True, exist_ok=True)

    mat = build_aligned_matrix(hrv_root, segment, emotibits, feature=feature, min_subjects=min_subjects)

    if mat.empty or mat.shape[1] < 2:
        summary = {
            "segment": segment,
            "status": "not_enough_data",
            "feature": feature,
            "n_subjects": int(mat.shape[1]) if not mat.empty else 0,
            "n_seconds": int(len(mat)) if not mat.empty else 0,
        }
        (seg_out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # Save aligned matrix
    aligned_path = seg_out / f"aligned_{feature}_matrix.csv"
    mat.to_csv(aligned_path, index_label="t_sec")

    # Zero-lag corr
    corr0 = mat.corr(method="pearson", min_periods=min_periods)
    corr0_path = seg_out / f"pairwise_corr_{feature}_zerolag.csv"
    corr0.to_csv(corr0_path)
    plot_heatmap(corr0, seg_out / f"pairwise_corr_{feature}_zerolag_heatmap.png",
                 title=f"{segment}: {feature} synchrony (zero-lag)", vmin=-1, vmax=1)

    # Metrics
    mpr0 = mean_pairwise_corr(mat, min_periods=min_periods)
    isc = leave_one_out_isc(mat, min_points=min_periods)

    best_r_mean = float("nan")
    mean_abs_lag = float("nan")

    # Best-lag outputs (optional)
    bestR_path = None
    bestL_path = None
    if max_lag_s and max_lag_s > 0:
        best_r_mean, mean_abs_lag = mean_pairwise_bestlag(mat, max_lag_s, min_points=min_periods)

        bestR, bestL = bestlag_corr_matrix(mat, max_lag_s, min_points=min_periods)

        bestR_path = seg_out / f"pairwise_corr_bestlag_pm{max_lag_s}.csv"
        bestL_path = seg_out / f"pairwise_bestlag_lag_s_pm{max_lag_s}.csv"
        bestR.to_csv(bestR_path)
        bestL.to_csv(bestL_path)

        plot_heatmap(bestR, seg_out / f"pairwise_corr_bestlag_pm{max_lag_s}_heatmap.png",
                     title=f"{segment}: {feature} best corr within ±{max_lag_s}s", vmin=-1, vmax=1)
        plot_heatmap(bestL, seg_out / f"pairwise_bestlag_lag_s_pm{max_lag_s}_heatmap.png",
                     title=f"{segment}: {feature} lag (s) at best corr (±{max_lag_s}s)")

    # Windowed synchrony (zero-lag)
    sync_ts = sliding_window_synchrony(mat, window_s=window_s, step_s=step_s, min_periods=min_periods)
    sync_csv = seg_out / f"synchrony_timeseries_{feature}.csv"
    sync_png = seg_out / f"synchrony_timeseries_{feature}.png"
    sync_ts.to_csv(sync_csv, index=False)
    plot_windowed_sync(sync_ts, sync_png,
                       title=f"{segment}: {feature} windowed synchrony (w={window_s}s, step={step_s}s)")

    summary = {
        "segment": segment,
        "status": "ok",
        "feature": feature,
        "n_subjects": int(mat.shape[1]),
        "n_seconds": int(len(mat)),
        "t_start_sec": int(mat.index.min()),
        "t_end_sec": int(mat.index.max()),
        "mean_pairwise_r_zero_lag": float(mpr0),
        "leave_one_out_isc": float(isc),
        "mean_pairwise_best_r_maxlag": float(best_r_mean),
        "mean_abs_lag_s_at_best_r": float(mean_abs_lag),
        "files": {
            "aligned_matrix": str(aligned_path),
            "pairwise_corr_zero_lag": str(corr0_path),
            "heatmap_zero_lag": str(seg_out / f"pairwise_corr_{feature}_zerolag_heatmap.png"),
            "windowed_sync_csv": str(sync_csv),
            "windowed_sync_png": str(sync_png),
            "pairwise_corr_bestlag": str(bestR_path) if bestR_path else None,
            "pairwise_lag_bestlag": str(bestL_path) if bestL_path else None,
        },
    }
    (seg_out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrv_root", default="analysis_outputs/hrv_v2",
                    help="Root folder containing EmotiBit_*/<segment>/HRV_1Hz.csv")
    ap.add_argument("--out_root", default="analysis_outputs/hrv_synchrony_v2",
                    help="Where to save synchrony outputs")

    ap.add_argument("--segments", default="baseline_warmup,meditation_2A,meditation_2B",
                    help="Comma-separated segments")

    ap.add_argument("--feature", default="rmssd_ms",
                    help="Which HRV feature to use: rmssd_ms or sdnn_ms")

    ap.add_argument("--window_s", type=int, default=30)
    ap.add_argument("--step_s", type=int, default=1)
    ap.add_argument("--max_lag_s", type=int, default=0)

    ap.add_argument("--min_subjects", type=int, default=5,
                    help="Keep seconds where at least this many subjects have data (default 5)")
    ap.add_argument("--min_periods", type=int, default=10,
                    help="Min overlapping points needed to compute correlations (default 10)")

    args = ap.parse_args()

    hrv_root = Path(args.hrv_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    emotibits = find_emotibit_ids(hrv_root)
    if not emotibits:
        raise SystemExit(f"No EmotiBit_* folders found under {hrv_root}")

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]

    summaries = []
    for seg in segments:
        summ = process_segment(
            hrv_root=hrv_root,
            out_root=out_root,
            segment=seg,
            emotibits=emotibits,
            feature=args.feature,
            window_s=int(args.window_s),
            step_s=int(args.step_s),
            max_lag_s=int(args.max_lag_s),
            min_subjects=int(args.min_subjects),
            min_periods=int(args.min_periods),
        )
        summaries.append(summ)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_root / f"synchrony_summary_{args.feature}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved HRV synchrony outputs under: {out_root}")
    print(f"Saved summary: {summary_path}")

    show_cols = ["segment", "status", "feature", "n_subjects", "n_seconds",
                 "mean_pairwise_r_zero_lag", "leave_one_out_isc"]
    if args.max_lag_s and args.max_lag_s > 0:
        show_cols += ["mean_pairwise_best_r_maxlag", "mean_abs_lag_s_at_best_r"]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""
runs:

RMSSD (zero-lag):
python3 analysisV2/04_hrv_synchrony.py \
  --hrv_root analysis_outputs/hrv_v2 \
  --out_root analysis_outputs/hrv_sync_v2_rmssd \
  --segments baseline_warmup,meditation_2A,meditation_2B \
  --feature rmssd_ms \
  --window_s 30 --step_s 1

RMSSD (±5s best-lag):
python3 analysisV2/04_hrv_synchrony.py \
  --hrv_root analysis_outputs/hrv_v2 \
  --out_root analysis_outputs/hrv_sync_v2_rmssd_lag5 \
  --segments baseline_warmup,meditation_2A,meditation_2B \
  --feature rmssd_ms \
  --window_s 30 --step_s 1 \
  --max_lag_s 5

SDNN (±5s best-lag):
python3 analysisV2/04_hrv_synchrony.py \
  --hrv_root analysis_outputs/hrv_v2 \
  --out_root analysis_outputs/hrv_sync_v2_sdnn_lag5 \
  --segments baseline_warmup,meditation_2A,meditation_2B \
  --feature sdnn_ms \
  --window_s 30 --step_s 1 \
  --max_lag_s 5
"""