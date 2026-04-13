# analysis_sync/code/02_isc_subject_significance_fdr.py
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# IO helpers (same as your pipeline)
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

    if "is_valid" in df.columns:
        is_valid = df["is_valid"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        out = out[is_valid]

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "hr"])
    out["time"] = out["time"].astype(float)
    out["hr"] = out["hr"].astype(float)
    return out


def align_to_seconds(df: pd.DataFrame) -> pd.Series:
    """Round timestamps to nearest second and take median per second."""
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
    """Z-score each subject column separately (within this matrix)."""
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
    """Stack matrices vertically; keep rows with >= min_subjects non-NaN."""
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

    return pd.concat(mats2, axis=0, ignore_index=True)


# ---------------------------
# ISC per subject
# ---------------------------
def isc_per_subject(mat: pd.DataFrame, min_points: int = 30) -> Dict[str, float]:
    """
    For each subject i: corr(x_i, mean(others)).
    Returns dict: subject -> ISC_i
    """
    if mat.empty or mat.shape[1] < 2:
        return {}

    cols = list(mat.columns)
    X = mat.to_numpy(dtype=float)
    out = {}

    for j, subj in enumerate(cols):
        x = X[:, j]
        others = np.delete(X, j, axis=1)
        y = np.nanmean(others, axis=1)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < int(min_points):
            out[subj] = np.nan
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        out[subj] = float(r) if np.isfinite(r) else np.nan

    return out


# ---------------------------
# Permutation: circular shifts (break alignment, keep autocorr)
# ---------------------------
def circular_shift_cols(mat: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
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


# ---------------------------
# FDR (Benjamini–Hochberg)
# ---------------------------
def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.
    Returns q-values aligned with input order.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]

    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev

    qvals = np.empty(n, dtype=float)
    qvals[order] = q
    return qvals


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_root", default="analysis_outputs/hr_clean_v2")
    ap.add_argument("--out_root", default="analysis_sync/outputs_subject_isc")
    ap.add_argument("--baseline_segment", default="baseline_warmup")
    ap.add_argument("--med_segments", default="meditation_2A,meditation_2B")
    ap.add_argument("--min_subjects", type=int, default=5)
    ap.add_argument("--min_points", type=int, default=30)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR threshold")
    ap.add_argument("--save_null_example", default="EmotiBit_1",
                    help="Which subject to save example null histogram for")
    args = ap.parse_args()

    hr_root = Path(args.hr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    emotibits = find_emotibit_ids(hr_root)
    if not emotibits:
        raise SystemExit(f"No EmotiBit_* folders found under {hr_root}")

    baseline_seg = args.baseline_segment.strip()
    med_segs = [s.strip() for s in args.med_segments.split(",") if s.strip()]

    # build matrices
    base_mat = build_segment_matrix(hr_root, baseline_seg, emotibits, min_subjects=args.min_subjects)
    base_z = zscore_within_subject(base_mat) if not base_mat.empty else pd.DataFrame()

    med_mats_z = []
    for seg in med_segs:
        m = build_segment_matrix(hr_root, seg, emotibits, min_subjects=args.min_subjects)
        med_mats_z.append(zscore_within_subject(m) if not m.empty else pd.DataFrame())

    base_concat = concat_mats([base_z], min_subjects=args.min_subjects)
    med_concat = concat_mats(med_mats_z, min_subjects=args.min_subjects)

    if base_concat.empty or med_concat.empty:
        raise SystemExit("Not enough data after alignment/filters (baseline or meditation concat is empty).")

    # observed ISC per subject
    isc_base = isc_per_subject(base_concat, min_points=args.min_points)
    isc_med = isc_per_subject(med_concat, min_points=args.min_points)

    subjects = sorted(set(isc_base.keys()) & set(isc_med.keys()))
    subjects = [s for s in subjects if np.isfinite(isc_base.get(s, np.nan)) and np.isfinite(isc_med.get(s, np.nan))]
    if len(subjects) < 2:
        raise SystemExit("Not enough valid subjects for ISC after min_points filtering.")

    obs_base = np.array([isc_base[s] for s in subjects], dtype=float)
    obs_med = np.array([isc_med[s] for s in subjects], dtype=float)
    obs_delta = obs_med - obs_base

    # permutation null for delta ISC per subject
    rng = np.random.default_rng(args.seed)
    null_delta = {s: [] for s in subjects}

    for _ in range(int(args.n_perm)):
        base_p = circular_shift_cols(base_concat, rng)
        med_p = circular_shift_cols(med_concat, rng)

        isc_b_p = isc_per_subject(base_p, min_points=args.min_points)
        isc_m_p = isc_per_subject(med_p, min_points=args.min_points)

        for s in subjects:
            vb = isc_b_p.get(s, np.nan)
            vm = isc_m_p.get(s, np.nan)
            if np.isfinite(vb) and np.isfinite(vm):
                null_delta[s].append(vm - vb)

    # p-values: one-sided (med > base) for each subject delta
    pvals = []
    null_means = []
    null_stds = []
    z_effects = []
    ci_low = []
    ci_high = []

    for i, s in enumerate(subjects):
        nd = np.asarray(null_delta[s], dtype=float)
        if len(nd) < 50:
            # too few perms survived for that subject
            pvals.append(np.nan)
            null_means.append(np.nan)
            null_stds.append(np.nan)
            z_effects.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            continue

        # one-sided p with +1 smoothing
        p = (np.sum(nd >= obs_delta[i]) + 1) / (len(nd) + 1)
        pvals.append(float(p))

        m = float(np.mean(nd))
        sd = float(np.std(nd)) if float(np.std(nd)) > 1e-8 else np.nan
        null_means.append(m)
        null_stds.append(sd)
        z_effects.append(float((obs_delta[i] - m) / sd) if np.isfinite(sd) else np.nan)

        # CI95 by shifting null quantiles around observed, similar to your group script
        ql, qh = np.percentile(nd, [2.5, 97.5])
        ci_low.append(float(obs_delta[i] + (ql - m)))
        ci_high.append(float(obs_delta[i] + (qh - m)))

    pvals_arr = np.asarray(pvals, dtype=float)

    # FDR across subjects (ignore NaNs)
    ok = np.isfinite(pvals_arr)
    qvals = np.full_like(pvals_arr, np.nan, dtype=float)
    if ok.sum() > 0:
        qvals[ok] = fdr_bh(pvals_arr[ok])

    # Save table
    rows = []
    for i, s in enumerate(subjects):
        rows.append({
            "subject": s,
            "isc_base": float(obs_base[i]),
            "isc_med": float(obs_med[i]),
            "delta_isc": float(obs_delta[i]),
            "p_one_sided_med_gt_base": float(pvals_arr[i]) if np.isfinite(pvals_arr[i]) else np.nan,
            "q_fdr_bh": float(qvals[i]) if np.isfinite(qvals[i]) else np.nan,
            "null_delta_mean": float(null_means[i]) if np.isfinite(null_means[i]) else np.nan,
            "null_delta_std": float(null_stds[i]) if np.isfinite(null_stds[i]) else np.nan,
            "delta_z_vs_null": float(z_effects[i]) if np.isfinite(z_effects[i]) else np.nan,
            "delta_ci95_low": float(ci_low[i]) if np.isfinite(ci_low[i]) else np.nan,
            "delta_ci95_high": float(ci_high[i]) if np.isfinite(ci_high[i]) else np.nan,
        })

    df_out = pd.DataFrame(rows).sort_values("subject")
    out_csv = out_root / "isc_subject_level_results.csv"
    df_out.to_csv(out_csv, index=False)

    # Summary JSON
    alpha = float(args.alpha)
    n_sig_unc = int(np.sum((pvals_arr < alpha) & np.isfinite(pvals_arr)))
    n_sig_fdr = int(np.sum((qvals < alpha) & np.isfinite(qvals)))

    out_json = out_root / "isc_subject_level_results.json"
    out_json.write_text(json.dumps({
        "baseline_segment": baseline_seg,
        "med_segments_combined": "|".join(med_segs),
        "min_subjects": int(args.min_subjects),
        "min_points": int(args.min_points),
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
        "n_subjects_used": int(len(subjects)),
        "alpha": alpha,
        "n_sig_uncorrected_p": n_sig_unc,
        "n_sig_fdr_q": n_sig_fdr,
        "files": {
            "csv": str(out_csv),
        }
    }, indent=2))

    print("Saved:", out_csv)
    print("Saved:", out_json)
    print(f"Subjects used: {len(subjects)}")
    print(f"Significant (p<{alpha}): {n_sig_unc} / {len(subjects)}")
    print(f"Significant (FDR q<{alpha}): {n_sig_fdr} / {len(subjects)}")

    # Plot: delta ISC per subject with q-values in title
    plt.figure(figsize=(9, 4))
    x = np.arange(len(subjects))
    plt.bar(x, obs_delta)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, subjects, rotation=45, ha="right")
    plt.ylabel("ΔISC = ISC_med - ISC_base")
    plt.title("Subject-level ΔISC (within-subject z); FDR applied across subjects")
    plt.tight_layout()
    out_png = out_root / "isc_subject_level_delta_bar.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved:", out_png)

    # Save one example null histogram for a chosen subject
    ex = args.save_null_example.strip()
    if ex in null_delta and len(null_delta[ex]) > 0:
        nd = np.asarray(null_delta[ex], dtype=float)
        obs = float(df_out[df_out["subject"] == ex]["delta_isc"].iloc[0])

        plt.figure(figsize=(7, 4))
        plt.hist(nd, bins=40)
        plt.axvline(obs, linewidth=2)
        plt.title(f"Null ΔISC (circular-shift) for {ex}")
        plt.xlabel("ΔISC (med - base) under null")
        plt.ylabel("count")
        plt.tight_layout()
        out_ex = out_root / f"isc_subject_level_null_example_hist_{ex}.png"
        plt.savefig(out_ex, dpi=150)
        plt.close()
        print("Saved:", out_ex)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Example run:

python3 analysis_sync/code/02_isc_subject_significance_fdr.py \
  --hr_root analysis_outputs/hr_clean \
  --out_root analysis_sync/outputs_subject_isc \
  --baseline_segment baseline_warmup \
  --med_segments meditation_all \
  --min_subjects 5 \
  --min_points 30 \
  --n_perm 2000 \
  --seed 0 \
  --alpha 0.05
"""