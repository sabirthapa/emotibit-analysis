# analysis_sync/code/03_hr_windowed_synchrony_permtest.py  (PASTABLE FULL FILE)
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# IO helpers
# ---------------------------
def load_hr_clean_csv(path: Path) -> pd.DataFrame:
    """
    Loads HR_1Hz_clean.csv files. Works with either:
      - columns: time, hr_bpm_clean, is_valid
      - or columns: time, hr_bpm, is_valid
    """
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {path}")

    if "hr_bpm_clean" in df.columns:
        hr_col = "hr_bpm_clean"
    elif "hr_bpm" in df.columns:
        hr_col = "hr_bpm"
    else:
        raise ValueError(f"Missing hr_bpm_clean/hr_bpm in {path}")

    out = df[["time", hr_col]].copy()
    out.rename(columns={hr_col: "hr"}, inplace=True)

    # respect is_valid if present
    if "is_valid" in df.columns:
        is_valid = df["is_valid"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        out = out[is_valid]

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "hr"])
    out["time"] = out["time"].astype(float)
    out["hr"] = out["hr"].astype(float)
    return out


def align_to_seconds(df: pd.DataFrame) -> pd.Series:
    """
    Round timestamps to nearest second and take median per second.
    Returns Series indexed by t_sec (int).
    """
    t_sec = np.rint(df["time"].to_numpy()).astype(np.int64)
    tmp = pd.DataFrame({"t_sec": t_sec, "hr": df["hr"].to_numpy(dtype=float)})
    return tmp.groupby("t_sec")["hr"].median().sort_index()


def find_emotibit_ids(hr_root: Path) -> List[str]:
    return sorted([p.name for p in hr_root.glob("EmotiBit_*") if p.is_dir()])


def hr_file(hr_root: Path, eb: str, segment: str) -> Path:
    return hr_root / eb / segment / "HR_1Hz_clean.csv"


# ---------------------------
# Matrix building
# ---------------------------
def build_segment_matrix(
    hr_root: Path,
    segment: str,
    emotibits: List[str],
    min_subjects: int,
) -> pd.DataFrame:
    """
    Build aligned HR matrix for ONE segment.
    Rows = integer seconds, Cols = subjects.
    Keeps rows where at least min_subjects have data.
    """
    series_map: Dict[str, pd.Series] = {}

    for eb in emotibits:
        f = hr_file(hr_root, eb, segment)
        if not f.exists():
            continue
        df = load_hr_clean_csv(f)
        s = align_to_seconds(df)
        if len(s) >= 10:
            series_map[eb] = s

    if len(series_map) < 2:
        return pd.DataFrame()

    # common overlap range (allow missing, enforce min_subjects per row)
    starts = [int(s.index.min()) for s in series_map.values()]
    ends = [int(s.index.max()) for s in series_map.values()]
    t0 = max(starts)
    t1 = min(ends)
    if t1 - t0 < 10:
        return pd.DataFrame()

    idx = pd.Index(np.arange(t0, t1 + 1, dtype=np.int64), name="t_sec")
    mat = pd.DataFrame({eb: series_map[eb].reindex(idx) for eb in sorted(series_map.keys())})

    keep = mat.notna().sum(axis=1) >= int(min_subjects)
    mat = mat.loc[keep]
    return mat


def zscore_within_subject(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each subject column separately (within this segment).
    """
    Z = mat.copy()
    for c in Z.columns:
        x = Z[c].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        Z[c] = (x - mu) / sd
    return Z


# ---------------------------
# Rolling-window synchrony
# ---------------------------
def pairwise_r_values(mat: pd.DataFrame, min_points: int) -> np.ndarray:
    """
    Pairwise Pearson r for all unique subject pairs.
    """
    if mat.shape[1] < 2:
        return np.asarray([], dtype=float)
    corr = mat.corr(method="pearson", min_periods=int(min_points))
    vals = corr.to_numpy()[np.triu_indices_from(corr.to_numpy(), k=1)]
    vals = vals[np.isfinite(vals)]
    return vals.astype(float)


def group_synchrony_from_window(mat_win: pd.DataFrame, min_points: int, agg: str) -> float:
    """
    Compute group synchrony in one window as:
      - agg='mean'   : mean of pairwise r
      - agg='median' : median of pairwise r  (robust, recommended)
    """
    vals = pairwise_r_values(mat_win, min_points=min_points)
    if len(vals) == 0:
        return float("nan")
    if agg == "median":
        return float(np.median(vals))
    return float(np.mean(vals))


def rolling_group_synchrony(
    mat: pd.DataFrame,
    window_s: int,
    step_s: int,
    min_points: int,
    agg: str,
) -> pd.DataFrame:
    """
    Rolling window group synchrony time series.
    Index of mat must be integer seconds (t_sec).
    Windows are [start, start+window_s).
    """
    if mat.empty:
        return pd.DataFrame(columns=["t_start", "t_end", "t_center", "sync"])

    t = mat.index.to_numpy(dtype=int)
    t0, t1 = int(t.min()), int(t.max())

    w = int(window_s)
    s = int(step_s)
    if w <= 1 or s <= 0:
        raise ValueError("window_s must be >1 and step_s must be >0")

    rows = []
    for start in range(t0, t1 - w + 1, s):
        end = start + w
        seg = mat.loc[(mat.index >= start) & (mat.index < end)]
        if len(seg) < int(min_points):
            rows.append({"t_start": start, "t_end": end, "t_center": start + w / 2.0, "sync": np.nan})
            continue
        r = group_synchrony_from_window(seg, min_points=min_points, agg=agg)
        rows.append({"t_start": start, "t_end": end, "t_center": start + w / 2.0, "sync": r})

    out = pd.DataFrame(rows)
    return out


# ---------------------------
# Permutation: circular shift within each segment
# ---------------------------
def circular_shift_cols(mat: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Circularly shift each subject column independently by a random offset.
    Preserves each subject’s autocorrelation but breaks across-person alignment.
    """
    X = mat.to_numpy(dtype=float)
    N, C = X.shape
    Y = np.empty_like(X)
    for j in range(C):
        x = X[:, j]
        if N <= 1:
            Y[:, j] = x
            continue
        k = int(rng.integers(0, N))  # 0..N-1
        Y[:, j] = np.roll(x, k)
    return pd.DataFrame(Y, index=mat.index, columns=mat.columns)


def summarize_sync_ts(df_ts: pd.DataFrame) -> Dict[str, float]:
    """
    Summaries for a rolling synchrony series.
    Uses finite sync values only.
    """
    x = df_ts["sync"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n_windows_used": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "n_windows_used": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
    }


def delta_ci_from_null(deltas: np.ndarray, delta_obs: float) -> Tuple[float, float]:
    """
    CI around delta_obs using null distribution width.
    shift quantiles by delta_obs - null_mean.
    """
    if deltas is None or len(deltas) == 0 or not np.isfinite(delta_obs):
        return float("nan"), float("nan")
    q_low, q_high = np.percentile(deltas, [2.5, 97.5])
    null_mean = float(np.mean(deltas))
    ci_low = delta_obs + (q_low - null_mean)
    ci_high = delta_obs + (q_high - null_mean)
    return float(ci_low), float(ci_high)


def z_vs_null(deltas: np.ndarray, delta_obs: float) -> float:
    if deltas is None or len(deltas) < 5 or not np.isfinite(delta_obs):
        return float("nan")
    mu = float(np.mean(deltas))
    sd = float(np.std(deltas))
    if not np.isfinite(sd) or sd < 1e-8:
        return float("nan")
    return float((delta_obs - mu) / sd)


# ---------------------------
# Plotting
# ---------------------------
def plot_sync_timeseries(
    base_ts: pd.DataFrame,
    med_ts: pd.DataFrame,
    out_png: Path,
    title: str,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    if not base_ts.empty:
        plt.plot(base_ts["t_center"], base_ts["sync"], label="baseline", linewidth=1.5)
    if not med_ts.empty:
        plt.plot(med_ts["t_center"], med_ts["sync"], label="meditation", linewidth=1.5)

    plt.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
    plt.title(title)
    plt.xlabel("time (sec, aligned)")
    plt.ylabel("rolling synchrony (pairwise r)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_sync_box(base_ts: pd.DataFrame, med_ts: pd.DataFrame, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xb = base_ts["sync"].to_numpy(dtype=float)
    xm = med_ts["sync"].to_numpy(dtype=float)
    xb = xb[np.isfinite(xb)]
    xm = xm[np.isfinite(xm)]

    plt.figure(figsize=(6, 4))
    plt.boxplot([xb, xm], labels=["baseline", "meditation"], showfliers=True)
    plt.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
    plt.title(title)
    plt.ylabel("rolling synchrony (pairwise r)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_null_hist(deltas: np.ndarray, delta_obs: float, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(deltas, bins=40)
    plt.axvline(delta_obs, linewidth=2)
    plt.title(title)
    plt.xlabel("delta = summary(med) - summary(base)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_root", default="analysis_outputs/hr_clean_v2",
                    help="Root with EmotiBit_*/<segment>/HR_1Hz_clean.csv")
    ap.add_argument("--out_root", default="analysis_sync/outputs_windowed",
                    help="Output folder for rolling-window synchrony test")

    ap.add_argument("--baseline_segment", default="baseline_warmup",
                    help="Baseline segment name")
    ap.add_argument("--med_segments", default="meditation_2A,meditation_2B",
                    help="Comma-separated meditation segments to combine (stack windows)")

    ap.add_argument("--min_subjects", type=int, default=5,
                    help="Keep timepoints with >= this many subjects")
    ap.add_argument("--min_points", type=int, default=30,
                    help="Min points inside a window for correlations (recommend 30)")

    ap.add_argument("--window_s", type=int, default=30,
                    help="Rolling window length in seconds (recommend 15/30/60)")
    ap.add_argument("--step_s", type=int, default=1,
                    help="Rolling window step in seconds (1 is fine)")
    ap.add_argument("--agg", default="median", choices=["median", "mean"],
                    help="How to aggregate pairwise r into a single group synchrony per window")

    ap.add_argument("--summary_stat", default="mean", choices=["mean", "median"],
                    help="Summary of rolling synchrony per condition used for delta test")

    ap.add_argument("--n_perm", type=int, default=2000,
                    help="Number of circular-shift permutations")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    hr_root = Path(args.hr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    emotibits = find_emotibit_ids(hr_root)
    if not emotibits:
        raise SystemExit(f"No EmotiBit_* folders found under {hr_root}")

    baseline_seg = args.baseline_segment.strip()
    med_segs = [s.strip() for s in args.med_segments.split(",") if s.strip()]
    if not med_segs:
        raise SystemExit("No meditation segments provided.")

    # --- Build matrices (per segment), then within-subject z-score ---
    base_mat = build_segment_matrix(hr_root, baseline_seg, emotibits, min_subjects=args.min_subjects)
    base_z = zscore_within_subject(base_mat) if not base_mat.empty else pd.DataFrame()

    med_mats_z = []
    for seg in med_segs:
        m = build_segment_matrix(hr_root, seg, emotibits, min_subjects=args.min_subjects)
        mz = zscore_within_subject(m) if not m.empty else pd.DataFrame()
        if not mz.empty:
            med_mats_z.append(mz)

    if base_z.empty or len(med_mats_z) == 0:
        raise SystemExit("Not enough data to run rolling-window synchrony (empty baseline or meditation matrices).")

    # --- Rolling synchrony time series ---
    base_ts = rolling_group_synchrony(
        base_z,
        window_s=args.window_s,
        step_s=args.step_s,
        min_points=args.min_points,
        agg=args.agg,
    )
    # meditation: compute per segment and then stack (keeps window meaning consistent)
    med_ts_parts = []
    for mz in med_mats_z:
        med_ts_parts.append(
            rolling_group_synchrony(
                mz,
                window_s=args.window_s,
                step_s=args.step_s,
                min_points=args.min_points,
                agg=args.agg,
            )
        )
    med_ts = pd.concat(med_ts_parts, axis=0, ignore_index=True)

    # Save TS CSVs
    out_base_ts = out_root / "rolling_sync_baseline.csv"
    out_med_ts = out_root / "rolling_sync_meditation.csv"
    base_ts.to_csv(out_base_ts, index=False)
    med_ts.to_csv(out_med_ts, index=False)

    # Summaries
    base_summ = summarize_sync_ts(base_ts)
    med_summ = summarize_sync_ts(med_ts)

    # Choose which summary statistic defines delta
    if args.summary_stat == "median":
        base_stat = base_summ["median"]
        med_stat = med_summ["median"]
    else:
        base_stat = base_summ["mean"]
        med_stat = med_summ["mean"]

    delta_obs = float(med_stat - base_stat)

    # --- Permutation test on delta (circular shift within each segment) ---
    rng = np.random.default_rng(args.seed)
    deltas = []

    for _ in range(int(args.n_perm)):
        # permute baseline
        base_perm = circular_shift_cols(base_z, rng)
        base_ts_p = rolling_group_synchrony(
            base_perm,
            window_s=args.window_s,
            step_s=args.step_s,
            min_points=args.min_points,
            agg=args.agg,
        )
        base_s_p = summarize_sync_ts(base_ts_p)
        base_stat_p = base_s_p["median"] if args.summary_stat == "median" else base_s_p["mean"]

        # permute each meditation segment separately, then stack
        med_stats = []
        for mz in med_mats_z:
            mz_p = circular_shift_cols(mz, rng)
            ts_p = rolling_group_synchrony(
                mz_p,
                window_s=args.window_s,
                step_s=args.step_s,
                min_points=args.min_points,
                agg=args.agg,
            )
            s_p = summarize_sync_ts(ts_p)
            med_stats.append(s_p["median"] if args.summary_stat == "median" else s_p["mean"])

        med_stat_p = float(np.mean(med_stats)) if len(med_stats) else float("nan")

        if np.isfinite(base_stat_p) and np.isfinite(med_stat_p):
            deltas.append(med_stat_p - base_stat_p)

    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        p_one = float("nan")
        p_two = float("nan")
        null_mu = float("nan")
        null_sd = float("nan")
        delta_z = float("nan")
        ci_low, ci_high = float("nan"), float("nan")
    else:
        p_one = (np.sum(deltas >= delta_obs) + 1) / (len(deltas) + 1)  # one-sided: med > base
        p_two = (np.sum(np.abs(deltas) >= abs(delta_obs)) + 1) / (len(deltas) + 1)
        null_mu = float(np.mean(deltas))
        null_sd = float(np.std(deltas))
        delta_z = z_vs_null(deltas, delta_obs)
        ci_low, ci_high = delta_ci_from_null(deltas, delta_obs)

    # --- Plots ---
    plot_sync_timeseries(
        base_ts, med_ts,
        out_png=out_root / "rolling_sync_timeseries_baseline_vs_meditation.png",
        title=f"Rolling synchrony ({args.agg} of pairwise r) | w={args.window_s}s step={args.step_s}s"
    )
    plot_sync_box(
        base_ts, med_ts,
        out_png=out_root / "rolling_sync_boxplot_baseline_vs_meditation.png",
        title="Distribution of rolling synchrony values"
    )
    if len(deltas) > 0 and np.isfinite(delta_obs):
        plot_null_hist(
            deltas,
            delta_obs,
            out_png=out_root / "null_hist_delta_rolling_med_minus_base.png",
            title="Null (circular-shift) for delta rolling synchrony: meditation - baseline",
        )

    # --- Save summary ---
    summary = {
        "baseline_segment": baseline_seg,
        "med_segments_combined": "|".join(med_segs),
        "n_subjects_total_found": int(len(emotibits)),
        "min_subjects_kept": int(args.min_subjects),
        "min_points": int(args.min_points),

        "window_s": int(args.window_s),
        "step_s": int(args.step_s),
        "window_agg": str(args.agg),
        "delta_summary_stat": str(args.summary_stat),

        "baseline_n_seconds_used": int(len(base_z)),
        "meditation_n_seconds_used_total": int(sum(len(m) for m in med_mats_z)),

        "baseline_rolling_summary": base_summ,
        "meditation_rolling_summary": med_summ,

        "baseline_stat_used": float(base_stat),
        "meditation_stat_used": float(med_stat),
        "delta_obs": float(delta_obs),

        "p_one_sided_med_gt_base": float(p_one),
        "p_two_sided": float(p_two),
        "n_null": int(len(deltas)),
        "null_delta_mean": float(null_mu),
        "null_delta_std": float(null_sd),
        "delta_z_vs_null": float(delta_z),
        "delta_ci95_low": float(ci_low),
        "delta_ci95_high": float(ci_high),

        "files": {
            "rolling_sync_baseline_csv": str(out_base_ts),
            "rolling_sync_meditation_csv": str(out_med_ts),
            "timeseries_png": str(out_root / "rolling_sync_timeseries_baseline_vs_meditation.png"),
            "boxplot_png": str(out_root / "rolling_sync_boxplot_baseline_vs_meditation.png"),
            "null_hist_png": str(out_root / "null_hist_delta_rolling_med_minus_base.png"),
        }
    }

    out_csv = out_root / "rolling_hr_synchrony_summary.csv"
    out_json = out_root / "rolling_hr_synchrony_summary.json"
    pd.DataFrame([{
        "baseline_segment": summary["baseline_segment"],
        "med_segments_combined": summary["med_segments_combined"],
        "window_s": summary["window_s"],
        "step_s": summary["step_s"],
        "agg": summary["window_agg"],
        "summary_stat": summary["delta_summary_stat"],
        "baseline_stat_used": summary["baseline_stat_used"],
        "meditation_stat_used": summary["meditation_stat_used"],
        "delta_obs": summary["delta_obs"],
        "p_one_sided_med_gt_base": summary["p_one_sided_med_gt_base"],
        "p_two_sided": summary["p_two_sided"],
        "delta_z_vs_null": summary["delta_z_vs_null"],
        "delta_ci95_low": summary["delta_ci95_low"],
        "delta_ci95_high": summary["delta_ci95_high"],
        "n_null": summary["n_null"],
    }]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("Saved plots under:", out_root)

    # Console summary (simple)
    print("\n=== ROLLING SYNCHRONY SUMMARY ===")
    print(f"Baseline rolling {args.summary_stat}:   {base_stat:.4f}")
    print(f"Meditation rolling {args.summary_stat}: {med_stat:.4f}")
    print(f"Delta (med-base): {delta_obs:.4f}")
    print(f"p(one-sided, med>base) = {p_one:.4f}")
    print(f"p(two-sided)          = {p_two:.4f}")
    print(f"delta_z_vs_null       = {delta_z:.2f}")
    print(f"delta CI95            = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"n_null                = {len(deltas)}")


if __name__ == "__main__":
    main()

"""
Example run (recommended):
python3 analysis_sync/code/03_hr_windowed_synchrony_permtest.py \
  --hr_root analysis_outputs/hr_clean \
  --out_root analysis_sync/outputs_windowed \
  --baseline_segment baseline_warmup \
  --med_segments meditation_all \
  --min_subjects 5 \
  --min_points 30 \
  --window_s 30 --step_s 1 \
  --agg median \
  --summary_stat mean \
  --n_perm 2000 \
  --seed 0
"""