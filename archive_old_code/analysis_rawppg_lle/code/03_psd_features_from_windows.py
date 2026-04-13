# analysis_rawppg_lle/code/03_psd_features_from_windows.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import signal as sig
except Exception as e:
    raise RuntimeError("This script needs scipy. Install: pip install scipy") from e


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def spectral_entropy_from_psd(psd_1d: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of normalized PSD (higher = more broadband/noisy)."""
    p = psd_1d.astype(float)
    p = np.clip(p, eps, None)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X_npy", default="analysis_rawppg_lle/outputs/rawppg_windows_X.npy")
    ap.add_argument("--meta_csv", default="analysis_rawppg_lle/outputs/rawppg_windows_meta.csv")
    ap.add_argument("--out_root", default="analysis_rawppg_lle/outputs/psd_features")

    # PSD settings
    ap.add_argument("--fs_nominal", type=float, default=100.0,
                    help="Use a single nominal fs for all windows (recommended here since windows were cut using fs_target=100).")
    ap.add_argument("--fmin", type=float, default=0.5)
    ap.add_argument("--fmax", type=float, default=8.0)
    ap.add_argument("--n_bins", type=int, default=64)

    # Welch settings
    ap.add_argument("--nperseg", type=int, default=256)
    ap.add_argument("--noverlap", type=int, default=128)
    ap.add_argument("--window", default="hann")

    args = ap.parse_args()

    X_path = Path(args.X_npy)
    meta_path = Path(args.meta_csv)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    X = np.load(X_path)  # [N, D]
    meta = pd.read_csv(meta_path)

    if len(meta) != X.shape[0]:
        raise RuntimeError(f"Meta rows ({len(meta)}) != X rows ({X.shape[0]})")

    # Safety cleanup
    X = X.astype(float)
    X[~np.isfinite(X)] = np.nan
    # If any NaNs still exist, fill per-row with row median
    if np.any(~np.isfinite(X)):
        row_med = np.nanmedian(X, axis=1)
        # replace NaNs with row median
        inds = np.where(~np.isfinite(X))
        X[inds] = row_med[inds[0]]

    print(f"Loaded X: {X_path} shape={X.shape}")
    print(f"Loaded meta: {meta_path} rows={len(meta)}")
    print(f"Welch PSD: fs={args.fs_nominal}, nperseg={args.nperseg}, noverlap={args.noverlap}, window={args.window}")
    print(f"Feature band: {args.fmin}–{args.fmax} Hz, bins={args.n_bins}")

    # Welch on ALL windows at once (axis=1)
    freqs, psd = sig.welch(
        X,
        fs=float(args.fs_nominal),
        window=str(args.window),
        nperseg=int(args.nperseg),
        noverlap=int(args.noverlap),
        detrend="constant",
        scaling="density",
        axis=1,
    )  # freqs: [F], psd: [N, F]

    band_mask = (freqs >= float(args.fmin)) & (freqs <= float(args.fmax))
    freqs_b = freqs[band_mask]
    psd_b = psd[:, band_mask]  # [N, Fb]

    # log-PSD for stability (avoid log(0))
    eps = 1e-12
    log_psd_b = np.log(psd_b + eps)

    # Bin edges across fmin..fmax
    edges = np.linspace(float(args.fmin), float(args.fmax), int(args.n_bins) + 1)

    # 64-bin features: mean(log PSD) per bin
    bin_feats = np.zeros((X.shape[0], int(args.n_bins)), dtype=float)
    for bi in range(int(args.n_bins)):
        a, b = edges[bi], edges[bi + 1]
        m = (freqs_b >= a) & (freqs_b < b) if bi < int(args.n_bins) - 1 else (freqs_b >= a) & (freqs_b <= b)
        if not np.any(m):
            bin_feats[:, bi] = 0.0
        else:
            bin_feats[:, bi] = np.mean(log_psd_b[:, m], axis=1)

    # Extra summary spectral features (computed on PSD, not log-PSD)
    # Total band power (integral PSD over band)
    total_power = np.trapz(psd_b, freqs_b, axis=1)

    # Peak frequency in the band (argmax PSD)
    peak_idx = np.argmax(psd_b, axis=1)
    peak_freq = freqs_b[peak_idx]

    # Bandpowers (integral) in sub-bands
    def bandpower(lo, hi):
        mm = (freqs_b >= lo) & (freqs_b <= hi)
        if not np.any(mm):
            return np.zeros((X.shape[0],), dtype=float)
        return np.trapz(psd_b[:, mm], freqs_b[mm], axis=1)

    bp_0p5_1 = bandpower(0.5, 1.0)
    bp_1_2 = bandpower(1.0, 2.0)
    bp_2_4 = bandpower(2.0, 4.0)
    bp_4_8 = bandpower(4.0, 8.0)

    # Spectral centroid (weighted average frequency)
    denom = np.sum(psd_b, axis=1) + eps
    centroid = np.sum(psd_b * freqs_b[None, :], axis=1) / denom

    # Spectral entropy (broadness)
    ent = np.array([spectral_entropy_from_psd(psd_b[i, :]) for i in range(psd_b.shape[0])], dtype=float)

    # Assemble output dataframe (meta + features)
    out = meta.copy()
    out["psd_total_power_0p5_8"] = total_power
    out["psd_peak_freq_0p5_8"] = peak_freq
    out["psd_bp_0p5_1"] = bp_0p5_1
    out["psd_bp_1_2"] = bp_1_2
    out["psd_bp_2_4"] = bp_2_4
    out["psd_bp_4_8"] = bp_4_8
    out["psd_centroid_0p5_8"] = centroid
    out["psd_entropy_0p5_8"] = ent

    for bi in range(int(args.n_bins)):
        out[f"psd_bin_{bi:02d}"] = bin_feats[:, bi]

    out_csv = out_root / "rawppg_psd_features.csv"
    out.to_csv(out_csv, index=False)

    # Save small summary
    summary_txt = out_root / "psd_features_summary.txt"
    summary_txt.write_text("\n".join([
        f"X_npy: {X_path}",
        f"meta_csv: {meta_path}",
        f"n_windows: {X.shape[0]}",
        f"window_dim: {X.shape[1]}",
        f"fs_nominal: {args.fs_nominal}",
        f"welch: nperseg={args.nperseg}, noverlap={args.noverlap}, window={args.window}",
        f"band: {args.fmin}–{args.fmax} Hz",
        f"bins: {args.n_bins}",
        "features: psd_total_power_0p5_8, psd_peak_freq_0p5_8, psd_bp_0p5_1, psd_bp_1_2, psd_bp_2_4, psd_bp_4_8, psd_centroid_0p5_8, psd_entropy_0p5_8, psd_bin_00..",
        f"saved_csv: {out_csv}",
    ]))

    print(f"\nSaved features CSV: {out_csv}")
    print(f"Saved summary:      {summary_txt}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
python3 analysis_rawppg_lle/code/03_psd_features_from_windows.py \
  --X_npy analysis_rawppg_lle/outputs/rawppg_windows_X.npy \
  --meta_csv analysis_rawppg_lle/outputs/rawppg_windows_meta.csv \
  --out_root analysis_rawppg_lle/outputs/psd_features \
  --fs_nominal 100 \
  --fmin 0.5 --fmax 8 \
  --n_bins 64 \
  --nperseg 256 --noverlap 128
"""