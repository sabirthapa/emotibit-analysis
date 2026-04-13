import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sig


def estimate_fs(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    return float(1.0 / dt)


def bandpass_ppg(x: np.ndarray, fs: float, low=0.7, high=4.0, order=4) -> np.ndarray:
    # 0.7–4.0 Hz ~ 42–240 BPM
    high = min(high, fs / 2 - 0.1)
    sos = sig.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    xf = sig.sosfiltfilt(sos, x)
    return xf


def detect_peaks(ppg_filt: np.ndarray, fs: float, hr_max=180) -> np.ndarray:
    # Min distance between peaks based on max HR
    min_dist = int(fs * (60.0 / hr_max))  # seconds per beat at hr_max
    min_dist = max(min_dist, 1)

    # Prominence relative to signal scale (robust-ish)
    prom = 0.5 * np.std(ppg_filt)
    if prom <= 0:
        prom = 1.0

    peaks, _ = sig.find_peaks(ppg_filt, distance=min_dist, prominence=prom)
    return peaks.astype(int)


def peaks_to_ibi_hr(t: np.ndarray, peaks: np.ndarray, hr_min=40, hr_max=180):
    if len(peaks) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    t_peaks = t[peaks]
    ibi = np.diff(t_peaks)  # seconds
    t_ibi = t_peaks[1:]
    hr = 60.0 / ibi

    ok = np.isfinite(hr) & (hr >= hr_min) & (hr <= hr_max) & np.isfinite(t_ibi)
    return t_ibi[ok], ibi[ok], hr[ok], t_peaks


def make_integer_second_grid(t0: float, t1: float) -> np.ndarray:
    # Align everyone to the SAME second grid: ..., 63108, 63109, ...
    start = int(np.ceil(t0))
    end = int(np.floor(t1))
    if end < start:
        return np.array([], dtype=float)
    return np.arange(start, end + 1, 1.0, dtype=float)


def interp_to_grid(t_hr: np.ndarray, hr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(t_hr) < 2 or len(grid) == 0:
        return np.full_like(grid, np.nan, dtype=float)

    # Ensure strictly increasing unique times
    t_u, idx = np.unique(t_hr, return_index=True)
    hr_u = hr[idx]
    if len(t_u) < 2:
        return np.full_like(grid, np.nan, dtype=float)

    return np.interp(grid, t_u, hr_u, left=np.nan, right=np.nan)


def nearest_time_distance(t_ref: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    For each grid time, distance to nearest t_ref element.
    Used to decide if a grid HR value is trustworthy (is_valid).
    """
    if len(t_ref) == 0 or len(grid) == 0:
        return np.full_like(grid, np.inf, dtype=float)

    idx = np.searchsorted(t_ref, grid, side="left")
    idx0 = np.clip(idx - 1, 0, len(t_ref) - 1)
    idx1 = np.clip(idx, 0, len(t_ref) - 1)

    d0 = np.abs(grid - t_ref[idx0])
    d1 = np.abs(grid - t_ref[idx1])
    return np.minimum(d0, d1)


def process_ppg_csv(ppg_csv_path: str, out_dir: str, fs_out: float = 1.0, max_gap_s: float = 2.5):
    df = pd.read_csv(ppg_csv_path)
    if "time" not in df.columns or "ppg" not in df.columns:
        return {"status": "bad_columns", "peaks": 0, "valid_beats": 0}

    t = df["time"].to_numpy(dtype=float)
    x = df["ppg"].to_numpy(dtype=float)

    if len(t) < 500:
        return {"status": "too_short", "peaks": 0, "valid_beats": 0}

    fs = estimate_fs(t)

    # Filter + normalize a bit (helps peak detector)
    xf = bandpass_ppg(x, fs)
    xf = (xf - np.median(xf)) / (np.std(xf) + 1e-8)

    peaks = detect_peaks(xf, fs)
    t_hr, ibi, hr, t_peaks = peaks_to_ibi_hr(t, peaks)

    grid = make_integer_second_grid(t[0], t[-1])
    hr_grid = interp_to_grid(t_hr, hr, grid)

    # Valid if the nearest instantaneous-HR timestamp is close enough
    d_near = nearest_time_distance(t_hr, grid)
    is_valid = np.isfinite(hr_grid) & (d_near <= max_gap_s)

    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({"time": t_peaks}).to_csv(os.path.join(out_dir, "peaks.csv"), index=False)
    pd.DataFrame({"time": t_hr, "ibi_s": ibi, "hr_bpm": hr}).to_csv(os.path.join(out_dir, "IBI.csv"), index=False)
    pd.DataFrame(
        {"time": grid, "hr_bpm": hr_grid, "is_valid": is_valid}
    ).to_csv(os.path.join(out_dir, "HR_1Hz_raw.csv"), index=False)

    dur = float(t[-1] - t[0])
    valid_beats = int(len(hr))
    status = "ok" if valid_beats >= 10 and len(grid) >= 10 else "low_beats_or_short"

    return {
        "status": status,
        "fs_est": round(fs, 3),
        "duration_s": round(dur, 3),
        "peaks": int(len(peaks)),
        "valid_beats": valid_beats,
        "mean_hr": float(np.nanmean(hr_grid)) if np.any(np.isfinite(hr_grid)) else np.nan,
        "sd_hr": float(np.nanstd(hr_grid)) if np.any(np.isfinite(hr_grid)) else np.nan,
        "valid_seconds": int(np.sum(is_valid)),
        "total_seconds": int(len(grid)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_root", default="analysis_outputs/segmented",
                    help="Root produced by 02.1_export_segments.py")
    ap.add_argument("--out_root", default="analysis_outputs/hr_raw",
                    help="Where to save HR outputs")
    ap.add_argument("--fs_out", type=float, default=1.0,
                    help="Kept for compatibility; output grid is integer-second 1Hz")
    ap.add_argument("--max_gap_s", type=float, default=2.5,
                    help="Seconds: if nearest beat-derived HR is farther than this, mark invalid")
    args = ap.parse_args()

    seg_root = Path(args.seg_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for ppg_csv in seg_root.glob("EmotiBit_*/**/PPG.csv"):
        eb = ppg_csv.parts[-3]      # EmotiBit_X
        segment = ppg_csv.parts[-2] # segment name
        out_dir = out_root / eb / segment

        qc = process_ppg_csv(str(ppg_csv), str(out_dir),
                             fs_out=args.fs_out, max_gap_s=args.max_gap_s)
        qc.update({"emotibit": eb, "segment": segment, "ppg_csv": str(ppg_csv)})
        rows.append(qc)

    qc_df = pd.DataFrame(rows).sort_values(["emotibit", "segment"])
    qc_path = out_root / "HR_QC_summary.csv"
    qc_df.to_csv(qc_path, index=False)

    print(f"Saved HR outputs under: {out_root}")
    print(f"Saved QC summary: {qc_path}")
    print(qc_df[["emotibit","segment","status","valid_beats","valid_seconds","total_seconds","mean_hr","sd_hr"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
python3 analysis/03_ppg_hr_extraction.py \
  --seg_root analysis_outputs/segmented \
  --out_root analysis_outputs/hr_raw \
  --max_gap_s 2.5
"""