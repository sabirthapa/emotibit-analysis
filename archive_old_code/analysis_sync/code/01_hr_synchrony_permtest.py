# analysis_sync/code/01_hr_synchrony_permtest.py  (PASTABLE FULL FILE)
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

    # common overlap range (not strict intersection; allow missing with min_subjects)
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
    Z-score each subject column separately (within this segment/matrix).
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


def concat_mats(mats: List[pd.DataFrame], min_subjects: int) -> pd.DataFrame:
    """
    Concatenate multiple segment matrices vertically (stack rows).
    Keep only rows with >= min_subjects non-NaN.
    """
    if not mats:
        return pd.DataFrame()

    cols = sorted(set().union(*[set(m.columns) for m in mats if not m.empty]))
    mats2 = []
    for m in mats:
        if m.empty:
            continue
        mm = m.reindex(columns=cols)
        keep = mm.notna().sum(axis=1) >= int(min_subjects)
        mats2.append(mm.loc[keep])

    if not mats2:
        return pd.DataFrame()

    out = pd.concat(mats2, axis=0, ignore_index=True)
    return out


# ---------------------------
# Metrics (zero-lag)
# ---------------------------
def mean_pairwise_corr(mat: pd.DataFrame, min_points: int = 30) -> float:
    """
    Mean pairwise Pearson correlation across subjects (zero lag).
    """
    if mat.shape[1] < 2:
        return float("nan")
    corr = mat.corr(method="pearson", min_periods=int(min_points))
    vals = corr.to_numpy()
    triu = vals[np.triu_indices_from(vals, k=1)]
    triu = triu[np.isfinite(triu)]
    return float(np.mean(triu)) if len(triu) else float("nan")


def pairwise_corr_values(mat: pd.DataFrame, min_points: int = 30) -> np.ndarray:
    """
    Return array of pairwise Pearson r for all unique subject pairs (zero-lag).
    """
    if mat.shape[1] < 2:
        return np.array([], dtype=float)
    corr = mat.corr(method="pearson", min_periods=int(min_points))
    vals = corr.to_numpy()[np.triu_indices_from(corr.to_numpy(), k=1)]
    vals = vals[np.isfinite(vals)]
    return vals.astype(float)


def summarize_pairwise(vals: np.ndarray) -> Dict:
    if vals is None or len(vals) == 0:
        return {
            "n_pairs": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_pos": 0,
            "n_neg": 0,
        }
    return {
        "n_pairs": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n_pos": int(np.sum(vals > 0)),
        "n_neg": int(np.sum(vals < 0)),
    }


def leave_one_out_isc(mat: pd.DataFrame, min_points: int = 30) -> float:
    """
    ISC: corr(subject_i, mean(other_subjects)) averaged over subjects.
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
        if ok.sum() < int(min_points):
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else float("nan")


# ---------------------------
# Metrics (best lag within ±max_lag_s)
# ---------------------------
def _corr_pair_at_lag(x: np.ndarray, y: np.ndarray, lag: int, min_points: int) -> float:
    """
    Pearson correlation at a given integer lag.
    lag > 0 means y is shifted forward (y happens later).
    """
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
    if ok.sum() < int(min_points):
        return float("nan")
    r = np.corrcoef(xs[ok], ys[ok])[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def max_lag_corr_pair(x: np.ndarray, y: np.ndarray, max_lag_s: int, min_points: int) -> Tuple[float, int]:
    """
    Best Pearson correlation between x and y allowing lags in [-max_lag_s, +max_lag_s].
    Returns (best_r, best_lag).
    """
    best_r = -np.inf
    best_lag = 0
    for lag in range(-int(max_lag_s), int(max_lag_s) + 1):
        r = _corr_pair_at_lag(x, y, lag, min_points=min_points)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_lag = lag
    if not np.isfinite(best_r):
        return float("nan"), 0
    return float(best_r), int(best_lag)


def mean_pairwise_bestlag(mat: pd.DataFrame, max_lag_s: int, min_points: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    For each pair, compute best correlation within ±max_lag_s.
    Returns:
      mean(best_r), mean(abs(best_lag)), best_r_values, best_lag_values
    """
    if mat.shape[1] < 2:
        return float("nan"), float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float)

    arr = mat.to_numpy(dtype=float)
    best_rs = []
    best_lags = []

    for i in range(arr.shape[1]):
        for j in range(i + 1, arr.shape[1]):
            r, lag = max_lag_corr_pair(arr[:, i], arr[:, j], max_lag_s=max_lag_s, min_points=min_points)
            if np.isfinite(r):
                best_rs.append(r)
                best_lags.append(lag)

    best_rs = np.asarray(best_rs, dtype=float)
    best_lags = np.asarray(best_lags, dtype=float)

    mean_r = float(np.mean(best_rs)) if len(best_rs) else float("nan")
    mean_abs_lag = float(np.mean(np.abs(best_lags))) if len(best_lags) else float("nan")
    return mean_r, mean_abs_lag, best_rs, best_lags


# ---------------------------
# Permutation: circular shifts
# ---------------------------
def circular_shift_cols(mat: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Circularly shift each subject column independently by a random offset.
    Preserves each subject’s autocorrelation but breaks alignment between people.
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
    return pd.DataFrame(Y, columns=mat.columns)


def permutation_test_delta_zero_lag(
    base_mats_z: List[pd.DataFrame],
    med_mats_z: List[pd.DataFrame],
    min_subjects: int,
    min_points: int,
    n_perm: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)

    base_obs = mean_pairwise_corr(concat_mats(base_mats_z, min_subjects=min_subjects), min_points=min_points)
    med_obs = mean_pairwise_corr(concat_mats(med_mats_z, min_subjects=min_subjects), min_points=min_points)
    delta_obs = med_obs - base_obs

    deltas = []
    for _ in range(int(n_perm)):
        base_perm = [circular_shift_cols(m, rng) for m in base_mats_z if not m.empty]
        med_perm = [circular_shift_cols(m, rng) for m in med_mats_z if not m.empty]

        base_r = mean_pairwise_corr(concat_mats(base_perm, min_subjects=min_subjects), min_points=min_points)
        med_r = mean_pairwise_corr(concat_mats(med_perm, min_subjects=min_subjects), min_points=min_points)

        if np.isfinite(base_r) and np.isfinite(med_r):
            deltas.append(med_r - base_r)

    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        return {
            "base_obs": float(base_obs),
            "med_obs": float(med_obs),
            "delta_obs": float(delta_obs),
            "n_null": 0,
            "p_one_sided_med_gt_base": float("nan"),
            "p_two_sided": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "null_deltas": np.asarray([], dtype=float),
        }

    p_one = (np.sum(deltas >= delta_obs) + 1) / (len(deltas) + 1)
    p_two = (np.sum(np.abs(deltas) >= abs(delta_obs)) + 1) / (len(deltas) + 1)

    return {
        "base_obs": float(base_obs),
        "med_obs": float(med_obs),
        "delta_obs": float(delta_obs),
        "n_null": int(len(deltas)),
        "p_one_sided_med_gt_base": float(p_one),
        "p_two_sided": float(p_two),
        "null_mean": float(np.mean(deltas)),
        "null_std": float(np.std(deltas)),
        "null_deltas": deltas,
    }


def permutation_test_delta_bestlag(
    base_mats_z: List[pd.DataFrame],
    med_mats_z: List[pd.DataFrame],
    min_subjects: int,
    min_points: int,
    max_lag_s: int,
    n_perm: int,
    seed: int,
) -> Dict:
    """
    Same circular-shift null, but the metric is mean pairwise BEST-LAG correlation within ±max_lag_s.
    """
    rng = np.random.default_rng(seed)

    base_concat = concat_mats(base_mats_z, min_subjects=min_subjects)
    med_concat = concat_mats(med_mats_z, min_subjects=min_subjects)

    base_obs, base_abs_lag, _, _ = mean_pairwise_bestlag(base_concat, max_lag_s=max_lag_s, min_points=min_points)
    med_obs, med_abs_lag, _, _ = mean_pairwise_bestlag(med_concat, max_lag_s=max_lag_s, min_points=min_points)
    delta_obs = med_obs - base_obs

    deltas = []
    abs_lags_base = []
    abs_lags_med = []

    for _ in range(int(n_perm)):
        base_perm = [circular_shift_cols(m, rng) for m in base_mats_z if not m.empty]
        med_perm = [circular_shift_cols(m, rng) for m in med_mats_z if not m.empty]

        b = concat_mats(base_perm, min_subjects=min_subjects)
        m = concat_mats(med_perm, min_subjects=min_subjects)

        b_r, b_abs, _, _ = mean_pairwise_bestlag(b, max_lag_s=max_lag_s, min_points=min_points)
        m_r, m_abs, _, _ = mean_pairwise_bestlag(m, max_lag_s=max_lag_s, min_points=min_points)

        if np.isfinite(b_r) and np.isfinite(m_r):
            deltas.append(m_r - b_r)
            abs_lags_base.append(b_abs)
            abs_lags_med.append(m_abs)

    deltas = np.asarray(deltas, dtype=float)
    abs_lags_base = np.asarray(abs_lags_base, dtype=float)
    abs_lags_med = np.asarray(abs_lags_med, dtype=float)

    if len(deltas) == 0:
        return {
            "base_obs": float(base_obs),
            "med_obs": float(med_obs),
            "delta_obs": float(delta_obs),
            "base_mean_abs_lag": float(base_abs_lag),
            "med_mean_abs_lag": float(med_abs_lag),
            "n_null": 0,
            "p_one_sided_med_gt_base": float("nan"),
            "p_two_sided": float("nan"),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "null_deltas": np.asarray([], dtype=float),
        }

    p_one = (np.sum(deltas >= delta_obs) + 1) / (len(deltas) + 1)
    p_two = (np.sum(np.abs(deltas) >= abs(delta_obs)) + 1) / (len(deltas) + 1)

    return {
        "base_obs": float(base_obs),
        "med_obs": float(med_obs),
        "delta_obs": float(delta_obs),
        "base_mean_abs_lag": float(base_abs_lag),
        "med_mean_abs_lag": float(med_abs_lag),
        "n_null": int(len(deltas)),
        "p_one_sided_med_gt_base": float(p_one),
        "p_two_sided": float(p_two),
        "null_mean": float(np.mean(deltas)),
        "null_std": float(np.std(deltas)),
        "null_deltas": deltas,
    }


def ci_from_null(deltas: np.ndarray, delta_obs: float) -> Dict:
    """
    CI around delta_obs using null distribution width.
    shift quantiles by delta_obs - null_mean.
    """
    if deltas is None or len(deltas) == 0 or not np.isfinite(delta_obs):
        return {"ci95_low": float("nan"), "ci95_high": float("nan")}
    q_low, q_high = np.percentile(deltas, [2.5, 97.5])
    null_mean = float(np.mean(deltas))
    ci_low = delta_obs + (q_low - null_mean)
    ci_high = delta_obs + (q_high - null_mean)
    return {"ci95_low": float(ci_low), "ci95_high": float(ci_high)}


def z_vs_null(deltas: np.ndarray, delta_obs: float) -> float:
    if deltas is None or len(deltas) < 5 or not np.isfinite(delta_obs):
        return float("nan")
    mu = float(np.mean(deltas))
    sd = float(np.std(deltas))
    if not np.isfinite(sd) or sd < 1e-8:
        return float("nan")
    return float((delta_obs - mu) / sd)


def plot_null_hist(deltas: np.ndarray, delta_obs: float, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(deltas, bins=40)
    plt.axvline(delta_obs, linewidth=2)
    plt.title(title)
    plt.xlabel("delta (med - base)")
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
    ap.add_argument("--out_root", default="analysis_sync/outputs",
                    help="Output folder for this synchrony test")

    ap.add_argument("--baseline_segment", default="baseline_warmup",
                    help="Baseline segment name")
    ap.add_argument("--med_segments", default="meditation_2A,meditation_2B",
                    help="Comma-separated meditation segments to combine")

    ap.add_argument("--min_subjects", type=int, default=5,
                    help="Keep timepoints with >= this many subjects")
    ap.add_argument("--min_points", type=int, default=30,
                    help="Min overlap points for correlation computations")

    ap.add_argument("--n_perm", type=int, default=2000,
                    help="Number of circular-shift permutations")
    ap.add_argument("--seed", type=int, default=0)

    # NEW: lagged synchrony
    ap.add_argument("--max_lag_s", type=int, default=0,
                    help="If >0, also compute best-lag synchrony within ±max_lag_s and permutation-test delta")

    args = ap.parse_args()

    hr_root = Path(args.hr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    emotibits = find_emotibit_ids(hr_root)
    if not emotibits:
        raise SystemExit(f"No EmotiBit_* folders found under {hr_root}")

    baseline_seg = args.baseline_segment.strip()
    med_segs = [s.strip() for s in args.med_segments.split(",") if s.strip()]

    # Build baseline matrix
    base_mat = build_segment_matrix(hr_root, baseline_seg, emotibits, min_subjects=args.min_subjects)
    base_z = zscore_within_subject(base_mat) if not base_mat.empty else pd.DataFrame()

    # Build meditation matrices and z-score each separately
    med_mats_z = []
    for seg in med_segs:
        m = build_segment_matrix(hr_root, seg, emotibits, min_subjects=args.min_subjects)
        mz = zscore_within_subject(m) if not m.empty else pd.DataFrame()
        med_mats_z.append(mz)

    base_concat = concat_mats([base_z], min_subjects=args.min_subjects)
    med_concat = concat_mats(med_mats_z, min_subjects=args.min_subjects)

    # --- zero-lag observed ---
    base_r = mean_pairwise_corr(base_concat, min_points=args.min_points)
    base_isc = leave_one_out_isc(base_concat, min_points=args.min_points)
    med_r = mean_pairwise_corr(med_concat, min_points=args.min_points)
    med_isc = leave_one_out_isc(med_concat, min_points=args.min_points)

    # Pairwise distributions (zero-lag)
    base_pair_vals = pairwise_corr_values(base_concat, min_points=args.min_points)
    med_pair_vals = pairwise_corr_values(med_concat, min_points=args.min_points)
    base_pair_summ = summarize_pairwise(base_pair_vals)
    med_pair_summ = summarize_pairwise(med_pair_vals)

    pair_base_csv = out_root / "pairwise_r_baseline.csv"
    pair_med_csv = out_root / "pairwise_r_meditation.csv"
    pd.DataFrame({"r": base_pair_vals}).to_csv(pair_base_csv, index=False)
    pd.DataFrame({"r": med_pair_vals}).to_csv(pair_med_csv, index=False)
    print("Saved:", pair_base_csv)
    print("Saved:", pair_med_csv)

    # --- permutation test (zero-lag delta) ---
    perm0 = permutation_test_delta_zero_lag(
        base_mats_z=[base_z],
        med_mats_z=med_mats_z,
        min_subjects=args.min_subjects,
        min_points=args.min_points,
        n_perm=args.n_perm,
        seed=args.seed,
    )
    deltas0 = perm0.get("null_deltas", np.asarray([], dtype=float))
    ci0 = ci_from_null(deltas0, perm0["delta_obs"])
    z0 = z_vs_null(deltas0, perm0["delta_obs"])

    # --- optional: lagged synchrony (best-lag within ±max_lag_s) ---
    lag_block = {}
    if int(args.max_lag_s) > 0:
        max_lag = int(args.max_lag_s)

        # observed best-lag means (on observed concatenated)
        base_best_r, base_abs_lag, base_best_rs, base_best_lags = mean_pairwise_bestlag(
            base_concat, max_lag_s=max_lag, min_points=args.min_points
        )
        med_best_r, med_abs_lag, med_best_rs, med_best_lags = mean_pairwise_bestlag(
            med_concat, max_lag_s=max_lag, min_points=args.min_points
        )

        # save pairwise bestlag values (nice to show)
        bestlag_base_csv = out_root / f"pairwise_bestlag_r_pm{max_lag}_baseline.csv"
        bestlag_med_csv = out_root / f"pairwise_bestlag_r_pm{max_lag}_meditation.csv"
        pd.DataFrame({"best_r": base_best_rs, "best_lag_s": base_best_lags}).to_csv(bestlag_base_csv, index=False)
        pd.DataFrame({"best_r": med_best_rs, "best_lag_s": med_best_lags}).to_csv(bestlag_med_csv, index=False)
        print("Saved:", bestlag_base_csv)
        print("Saved:", bestlag_med_csv)

        # permutation test for best-lag delta
        permlag = permutation_test_delta_bestlag(
            base_mats_z=[base_z],
            med_mats_z=med_mats_z,
            min_subjects=args.min_subjects,
            min_points=args.min_points,
            max_lag_s=max_lag,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        deltasL = permlag.get("null_deltas", np.asarray([], dtype=float))
        ciL = ci_from_null(deltasL, permlag["delta_obs"])
        zL = z_vs_null(deltasL, permlag["delta_obs"])

        lag_block = {
            "max_lag_s": max_lag,
            "baseline_mean_pairwise_bestlag_r": float(base_best_r),
            "baseline_mean_abs_bestlag_s": float(base_abs_lag),
            "med_mean_pairwise_bestlag_r": float(med_best_r),
            "med_mean_abs_bestlag_s": float(med_abs_lag),
            "delta_bestlag_r_obs": float(permlag["delta_obs"]),
            "p_bestlag_one_sided_med_gt_base": float(permlag["p_one_sided_med_gt_base"]),
            "p_bestlag_two_sided": float(permlag["p_two_sided"]),
            "bestlag_null_delta_mean": float(permlag["null_mean"]),
            "bestlag_null_delta_std": float(permlag["null_std"]),
            "bestlag_delta_z_vs_null": float(zL),
            "bestlag_delta_ci95_low": float(ciL["ci95_low"]),
            "bestlag_delta_ci95_high": float(ciL["ci95_high"]),
            "files_bestlag": {
                "pairwise_bestlag_baseline_csv": str(bestlag_base_csv),
                "pairwise_bestlag_meditation_csv": str(bestlag_med_csv),
            },
        }

        # plot histogram for bestlag delta
        if isinstance(deltasL, np.ndarray) and len(deltasL) > 0 and np.isfinite(permlag["delta_obs"]):
            out_png = out_root / f"null_hist_delta_bestlag_pm{max_lag}_med_minus_base.png"
            plot_null_hist(
                deltasL,
                permlag["delta_obs"],
                out_png,
                title=f"Null (circular-shift) for delta BEST-LAG synchrony (±{max_lag}s): meditation - baseline",
            )
            print("Saved:", out_png)

    # --- write summary ---
    summary = {
        "baseline_segment": baseline_seg,
        "med_segments_combined": "|".join(med_segs),
        "n_subjects_total_found": len(emotibits),
        "min_subjects_kept": int(args.min_subjects),
        "min_points": int(args.min_points),

        "baseline_n_seconds_used": int(len(base_concat)),
        "meditation_n_seconds_used": int(len(med_concat)),

        "baseline_mean_pairwise_r_z": float(base_r),
        "baseline_leave_one_out_isc_z": float(base_isc),

        "med_mean_pairwise_r_z": float(med_r),
        "med_leave_one_out_isc_z": float(med_isc),

        # pairwise distribution summaries (zero-lag)
        "baseline_pairwise_mean": float(base_pair_summ["mean"]),
        "baseline_pairwise_median": float(base_pair_summ["median"]),
        "baseline_pairwise_q25": float(base_pair_summ["q25"]),
        "baseline_pairwise_q75": float(base_pair_summ["q75"]),
        "baseline_pos_pairs": int(base_pair_summ["n_pos"]),
        "baseline_neg_pairs": int(base_pair_summ["n_neg"]),

        "med_pairwise_mean": float(med_pair_summ["mean"]),
        "med_pairwise_median": float(med_pair_summ["median"]),
        "med_pairwise_q25": float(med_pair_summ["q25"]),
        "med_pairwise_q75": float(med_pair_summ["q75"]),
        "med_pos_pairs": int(med_pair_summ["n_pos"]),
        "med_neg_pairs": int(med_pair_summ["n_neg"]),

        # delta test (zero-lag)
        "delta_r_obs": float(perm0["delta_obs"]),
        "p_one_sided_med_gt_base": float(perm0["p_one_sided_med_gt_base"]),
        "p_two_sided": float(perm0["p_two_sided"]),
        "n_null": int(perm0["n_null"]),
        "null_delta_mean": float(perm0["null_mean"]),
        "null_delta_std": float(perm0["null_std"]),
        "delta_r_z_vs_null": float(z0),
        "delta_r_ci95_low": float(ci0["ci95_low"]),
        "delta_r_ci95_high": float(ci0["ci95_high"]),

        # files
        "files": {
            "pairwise_r_baseline_csv": str(pair_base_csv),
            "pairwise_r_meditation_csv": str(pair_med_csv),
        }
    }

    if lag_block:
        summary.update(lag_block)

    out_csv = out_root / "hr_synchrony_baseline_vs_meditation.csv"
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    out_json = out_root / "hr_synchrony_baseline_vs_meditation.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)

    # plot null histogram (zero-lag delta)
    if isinstance(deltas0, np.ndarray) and len(deltas0) > 0 and np.isfinite(perm0["delta_obs"]):
        out_png = out_root / "null_hist_delta_med_minus_base.png"
        plot_null_hist(
            deltas0,
            perm0["delta_obs"],
            out_png,
            title="Null (circular-shift) for delta synchrony (ZERO-LAG): meditation - baseline",
        )
        print("Saved:", out_png)

    # console output
    print("\n=== OBSERVED (within-subject z) ===")
    print(f"Baseline:   mean_pairwise_r = {base_r:.4f},  ISC = {base_isc:.4f},  seconds used = {len(base_concat)}")
    print(f"Meditation: mean_pairwise_r = {med_r:.4f},  ISC = {med_isc:.4f},  seconds used = {len(med_concat)}")

    print("\n=== PAIRWISE DISTRIBUTION (r, zero-lag) ===")
    print("Baseline pairs:", base_pair_summ)
    print("Meditation pairs:", med_pair_summ)

    print("\n=== DELTA TEST (ZERO-LAG, med - base) ===")
    print(f"delta_r = {perm0['delta_obs']:.4f}")
    print(f"p(one-sided, med > base) = {perm0['p_one_sided_med_gt_base']:.4f}")
    print(f"p(two-sided)             = {perm0['p_two_sided']:.4f}")
    print(f"null mean±std            = {perm0['null_mean']:.4f} ± {perm0['null_std']:.4f}")
    print(f"delta_z_vs_null          = {z0:.2f}")
    print(f"delta CI95               = [{ci0['ci95_low']:.4f}, {ci0['ci95_high']:.4f}]")
    print(f"n_null                   = {perm0['n_null']}")

    if lag_block:
        print(f"\n=== LAGGED SYNCHRONY (BEST-LAG within ±{lag_block['max_lag_s']}s) ===")
        print(f"Baseline bestlag mean r  = {lag_block['baseline_mean_pairwise_bestlag_r']:.4f} (mean |lag|={lag_block['baseline_mean_abs_bestlag_s']:.2f}s)")
        print(f"Meditation bestlag mean r= {lag_block['med_mean_pairwise_bestlag_r']:.4f} (mean |lag|={lag_block['med_mean_abs_bestlag_s']:.2f}s)")
        print(f"delta_bestlag_r          = {lag_block['delta_bestlag_r_obs']:.4f}")
        print(f"p_bestlag(one-sided)     = {lag_block['p_bestlag_one_sided_med_gt_base']:.4f}")
        print(f"p_bestlag(two-sided)     = {lag_block['p_bestlag_two_sided']:.4f}")
        print(f"bestlag null mean±std    = {lag_block['bestlag_null_delta_mean']:.4f} ± {lag_block['bestlag_null_delta_std']:.4f}")
        print(f"bestlag delta_z_vs_null  = {lag_block['bestlag_delta_z_vs_null']:.2f}")
        print(f"bestlag delta CI95       = [{lag_block['bestlag_delta_ci95_low']:.4f}, {lag_block['bestlag_delta_ci95_high']:.4f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
(archive_old_code: baseline vs meditation_all)

ZERO-LAG only:
python3 analysis_sync/code/01_hr_synchrony_permtest.py \
  --hr_root analysis_outputs/hr_clean \
  --out_root analysis_sync/outputs \
  --baseline_segment baseline_warmup \
  --med_segments meditation_all \
  --min_subjects 5 \
  --min_points 30 \
  --n_perm 2000 \
  --seed 0

ZERO-LAG + BEST-LAG ±5s:
python3 analysis_sync/code/01_hr_synchrony_permtest.py \
  --hr_root analysis_outputs/hr_clean \
  --out_root analysis_sync/outputs \
  --baseline_segment baseline_warmup \
  --med_segments meditation_all \
  --min_subjects 5 \
  --min_points 30 \
  --n_perm 2000 \
  --seed 0 \
  --max_lag_s 5
"""