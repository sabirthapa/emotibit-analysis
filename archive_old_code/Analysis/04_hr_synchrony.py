# analysis/04_hr_synchrony.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_hr_csv(path: Path, use_clean: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {path}")

    # Choose HR column
    if use_clean and "hr_bpm_clean" in df.columns:
        hr_col = "hr_bpm_clean"
    elif "hr_bpm" in df.columns:
        hr_col = "hr_bpm"
    else:
        raise ValueError(f"Missing hr_bpm/hr_bpm_clean columns in {path}")

    out = df[["time", hr_col]].copy()
    out.rename(columns={hr_col: "hr"}, inplace=True)

    # Respect is_valid if present
    if "is_valid" in df.columns:
        is_valid = df["is_valid"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        out = out[is_valid]

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "hr"])
    out["time"] = out["time"].astype(float)
    out["hr"] = out["hr"].astype(float)
    return out


def align_to_seconds(df: pd.DataFrame) -> pd.Series:
    """
    Convert timestamps to integer seconds using rounding.
    Handles tiny ms differences across EmotiBits.
    Then collapse duplicates by median HR per second.
    """
    t_sec = np.rint(df["time"].to_numpy()).astype(np.int64)
    tmp = pd.DataFrame({"t_sec": t_sec, "hr": df["hr"].to_numpy()})
    return tmp.groupby("t_sec")["hr"].median().sort_index()


def find_emotibit_ids(hr_root: Path) -> List[str]:
    return sorted([p.name for p in hr_root.glob("EmotiBit_*") if p.is_dir()])


def segment_file(hr_root: Path, emotibit: str, segment: str) -> Path:
    return hr_root / emotibit / segment / "HR_1Hz_clean.csv"


def mean_pairwise_corr(mat: pd.DataFrame, min_periods: int = 10) -> float:
    """
    Mean Pearson correlation across all unique subject pairs at zero lag.
    mat: rows=time, cols=subjects
    """
    if mat.shape[1] < 2:
        return float("nan")

    corr = mat.corr(method="pearson", min_periods=min_periods)
    vals = corr.to_numpy()
    triu = vals[np.triu_indices_from(vals, k=1)]
    triu = triu[np.isfinite(triu)]
    return float(np.mean(triu)) if len(triu) else float("nan")


def leave_one_out_isc(mat: pd.DataFrame, min_points: int = 10) -> float:
    """
    ISC-style metric: for each subject i, correlate their HR with the mean HR of all other subjects.
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
    Best Pearson correlation between x and y allowing integer lags in [-max_lag_s, +max_lag_s].
    Lag > 0 means y is shifted forward (y happens later).
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


def mean_pairwise_maxlag(mat: pd.DataFrame, max_lag_s: int, min_points: int = 10) -> Tuple[float, float]:
    """
    For each pair, compute best correlation within +/- max_lag_s seconds.
    Returns:
      mean(best_r), mean(abs(best_lag))
    """
    if mat.shape[1] < 2:
        return float("nan"), float("nan")

    arr = mat.to_numpy(dtype=float)
    best_rs = []
    best_lags = []

    for i in range(arr.shape[1]):
        for j in range(i + 1, arr.shape[1]):
            r, lag = max_lag_corr_pair(arr[:, i], arr[:, j], max_lag_s, min_points=min_points)
            if np.isfinite(r):
                best_rs.append(r)
                best_lags.append(abs(lag))

    return (
        float(np.mean(best_rs)) if best_rs else float("nan"),
        float(np.mean(best_lags)) if best_lags else float("nan"),
    )


def bestlag_corr_and_lag_matrices(mat: pd.DataFrame, max_lag_s: int, min_points: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two NxN matrices:
      1) best-lag correlation matrix (Pearson r after allowing +/- max_lag_s shift)
      2) best-lag (signed) lag matrix in seconds
         lag[i,j] = lag that maximizes corr between i and j
         lag[j,i] = -lag[i,j]
    """
    cols = list(mat.columns)
    arr = mat.to_numpy(dtype=float)
    n = len(cols)

    corr_mat = np.eye(n, dtype=float)
    lag_mat = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            r, lag = max_lag_corr_pair(arr[:, i], arr[:, j], int(max_lag_s), min_points=min_points)
            corr_mat[i, j] = r
            corr_mat[j, i] = r
            lag_mat[i, j] = int(lag)
            lag_mat[j, i] = int(-lag)

    corr_df = pd.DataFrame(corr_mat, index=cols, columns=cols)
    lag_df = pd.DataFrame(lag_mat, index=cols, columns=cols)
    return corr_df, lag_df


def sliding_window_synchrony(mat: pd.DataFrame, window_s: int, step_s: int, min_periods: int = 10) -> pd.DataFrame:
    """
    Compute mean pairwise corr over sliding windows.
    Assumes mat index is integer seconds (t_sec).
    """
    if mat.empty:
        return pd.DataFrame(columns=["t_center", "mean_pairwise_r", "t_start", "t_end"])

    t = mat.index.to_numpy()
    t0, t1 = int(t.min()), int(t.max())

    rows = []
    w = int(window_s)
    s = int(step_s)
    if w <= 0 or s <= 0:
        raise ValueError("window_s and step_s must be > 0")

    for start in range(t0, t1 - w + 1, s):
        end = start + w
        seg = mat.loc[(mat.index >= start) & (mat.index < end)]
        r = mean_pairwise_corr(seg, min_periods=min_periods) if len(seg) >= min_periods else float("nan")
        rows.append({"t_center": start + w / 2.0, "mean_pairwise_r": r, "t_start": start, "t_end": end})

    return pd.DataFrame(rows)


def plot_corr_heatmap(corr: pd.DataFrame, out_png: Path, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_lag_heatmap(lag_df: pd.DataFrame, out_png: Path, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(lag_df.to_numpy(), aspect="auto")
    plt.colorbar(label="best lag (s)")
    plt.xticks(range(len(lag_df.columns)), lag_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(lag_df.index)), lag_df.index)
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


def build_aligned_matrix(
    hr_root: Path,
    segment: str,
    emotibits: List[str],
    use_clean: bool,
    min_subjects: int,
) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}

    for eb in emotibits:
        f = segment_file(hr_root, eb, segment)
        if not f.exists():
            continue
        df = load_hr_csv(f, use_clean=use_clean)
        s = align_to_seconds(df)
        if len(s) >= 10:
            series_map[eb] = s

    if len(series_map) < 2:
        return pd.DataFrame()

    # Common time range (overlap)
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


def process_segment(
    hr_root: Path,
    out_root: Path,
    segment: str,
    emotibits: List[str],
    use_clean: bool,
    window_s: int,
    step_s: int,
    max_lag_s: int,
    min_subjects: int,
    min_periods: int,
) -> Dict:
    seg_out = out_root / segment
    seg_out.mkdir(parents=True, exist_ok=True)

    mat = build_aligned_matrix(hr_root, segment, emotibits, use_clean=use_clean, min_subjects=min_subjects)

    if mat.empty or mat.shape[1] < 2:
        summary = {
            "segment": segment,
            "status": "not_enough_data",
            "n_subjects": int(mat.shape[1]) if not mat.empty else 0,
            "n_seconds": int(len(mat)) if not mat.empty else 0,
        }
        (seg_out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    aligned_path = seg_out / "aligned_HR_matrix.csv"
    mat.to_csv(aligned_path, index_label="t_sec")

    # --- Zero-lag corr ---
    corr0 = mat.corr(method="pearson", min_periods=min_periods)
    corr0.to_csv(seg_out / "pairwise_corr.csv")
    plot_corr_heatmap(
        corr0,
        seg_out / "pairwise_corr_heatmap.png",
        title=f"{segment}: pairwise HR synchrony (zero-lag)",
    )

    # --- Metrics ---
    mpr = mean_pairwise_corr(mat, min_periods=min_periods)
    isc = leave_one_out_isc(mat, min_points=min_periods)

    best_r = float("nan")
    mean_abs_lag = float("nan")

    # --- Best-lag outputs ---
    if max_lag_s and max_lag_s > 0:
        best_r, mean_abs_lag = mean_pairwise_maxlag(mat, int(max_lag_s), min_points=min_periods)

        corr_bl, lag_bl = bestlag_corr_and_lag_matrices(mat, int(max_lag_s), min_points=min_periods)

        corr_bl_path = seg_out / f"pairwise_corr_bestlag_pm{int(max_lag_s)}.csv"
        lag_bl_path = seg_out / f"pairwise_bestlag_lag_s_pm{int(max_lag_s)}.csv"
        corr_bl.to_csv(corr_bl_path)
        lag_bl.to_csv(lag_bl_path)

        plot_corr_heatmap(
            corr_bl,
            seg_out / f"pairwise_corr_bestlag_pm{int(max_lag_s)}_heatmap.png",
            title=f"{segment}: pairwise HR synchrony (best lag ±{int(max_lag_s)}s)",
        )
        plot_lag_heatmap(
            lag_bl,
            seg_out / f"pairwise_bestlag_lag_s_pm{int(max_lag_s)}_heatmap.png",
            title=f"{segment}: best lag per pair (±{int(max_lag_s)}s)",
        )

    # --- Sliding window synchrony ---
    sync_ts = sliding_window_synchrony(mat, window_s=int(window_s), step_s=int(step_s), min_periods=min_periods)
    sync_ts.to_csv(seg_out / "synchrony_timeseries.csv", index=False)
    plot_windowed_sync(
        sync_ts,
        seg_out / "synchrony_timeseries.png",
        title=f"{segment}: windowed synchrony (w={window_s}s, step={step_s}s)",
    )

    files = {
        "aligned_matrix": str(aligned_path),
        "pairwise_corr_zero_lag": str(seg_out / "pairwise_corr.csv"),
        "heatmap_zero_lag_png": str(seg_out / "pairwise_corr_heatmap.png"),
        "windowed_sync_csv": str(seg_out / "synchrony_timeseries.csv"),
        "windowed_sync_png": str(seg_out / "synchrony_timeseries.png"),
    }

    if max_lag_s and max_lag_s > 0:
        files.update({
            "pairwise_corr_bestlag": str(seg_out / f"pairwise_corr_bestlag_pm{int(max_lag_s)}.csv"),
            "pairwise_bestlag_lag_s": str(seg_out / f"pairwise_bestlag_lag_s_pm{int(max_lag_s)}.csv"),
            "heatmap_bestlag_png": str(seg_out / f"pairwise_corr_bestlag_pm{int(max_lag_s)}_heatmap.png"),
            "heatmap_lag_png": str(seg_out / f"pairwise_bestlag_lag_s_pm{int(max_lag_s)}_heatmap.png"),
        })

    summary = {
        "segment": segment,
        "status": "ok",
        "n_subjects": int(mat.shape[1]),
        "n_seconds": int(len(mat)),
        "t_start_sec": int(mat.index.min()),
        "t_end_sec": int(mat.index.max()),
        "mean_pairwise_r_zero_lag": float(mpr),
        "leave_one_out_isc": float(isc),
        "mean_pairwise_best_r_maxlag": float(best_r),
        "mean_abs_lag_s_at_best_r": float(mean_abs_lag),
        "max_lag_s": int(max_lag_s),
        "files": files,
    }
    (seg_out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_root", default="analysis_outputs/hr_clean",
                    help="Root folder containing EmotiBit_*/<segment>/HR_1Hz_clean.csv")
    ap.add_argument("--out_root", default="analysis_outputs/synchrony",
                    help="Where to save synchrony outputs")
    ap.add_argument("--segments", default="baseline_warmup,meditation_all",
                    help="Comma-separated segment names")

    # default uses clean; pass --use_raw if you want hr_bpm
    ap.add_argument("--use_raw", action="store_true",
                    help="Use hr_bpm instead of hr_bpm_clean")

    ap.add_argument("--window_s", type=int, default=30)
    ap.add_argument("--step_s", type=int, default=1)
    ap.add_argument("--max_lag_s", type=int, default=0)

    ap.add_argument("--min_subjects", type=int, default=5,
                    help="Keep seconds where at least this many subjects have data (default 5)")
    ap.add_argument("--min_periods", type=int, default=10,
                    help="Min overlapping points needed to compute correlations (default 10)")

    args = ap.parse_args()

    hr_root = Path(args.hr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    emotibits = find_emotibit_ids(hr_root)
    if not emotibits:
        raise SystemExit(f"No EmotiBit_* folders found under {hr_root}")

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    use_clean = not bool(args.use_raw)

    summaries = []
    for seg in segments:
        summ = process_segment(
            hr_root=hr_root,
            out_root=out_root,
            segment=seg,
            emotibits=emotibits,
            use_clean=use_clean,
            window_s=int(args.window_s),
            step_s=int(args.step_s),
            max_lag_s=int(args.max_lag_s),
            min_subjects=int(args.min_subjects),
            min_periods=int(args.min_periods),
        )
        summaries.append(summ)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_root / "synchrony_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved synchrony outputs under: {out_root}")
    print(f"Saved combined summary: {summary_path}")

    show_cols = ["segment", "status", "n_subjects", "n_seconds",
                 "mean_pairwise_r_zero_lag", "leave_one_out_isc"]
    if args.max_lag_s and args.max_lag_s > 0:
        show_cols += ["mean_pairwise_best_r_maxlag", "mean_abs_lag_s_at_best_r"]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""
clean HR:
python3 analysis/04_hr_synchrony.py \
  --hr_root analysis_outputs/hr_clean \
  --out_root analysis_outputs/synchrony \
  --segments baseline_warmup,meditation_all \
  --window_s 30 --step_s 1

With lag
python3 analysis/04_hr_synchrony.py \
  --hr_root analysis_outputs/hr_clean \
  --out_root analysis_outputs/synchrony_lag5 \
  --segments baseline_warmup,meditation_all \
  --window_s 30 --step_s 1 \
  --max_lag_s 5
"""