# analysis_rawppg_lle/code/01_extract_raw_windows.py
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyxdf

try:
    from scipy import signal as sig
except Exception:
    sig = None


# -----------------------------
# Marker helpers
# -----------------------------
def normalize_marker_label(label: str) -> str:
    lab = str(label).strip().lower()
    lab = re.sub(r"\s+", " ", lab)
    return lab.replace(" ", "_")


def find_marker_stream(streams: List[dict]) -> dict:
    for s in streams:
        name = str(s["info"]["name"][0]).lower()
        stype = str(s["info"]["type"][0]).lower()
        if "marker" in stype or "marker" in name:
            return s
    raise RuntimeError("No marker stream found (type/name contains 'marker').")


def extract_markers(marker_stream: dict) -> Tuple[np.ndarray, List[str]]:
    mt = np.asarray(marker_stream["time_stamps"], dtype=float)
    raw = marker_stream["time_series"]
    labels: List[str] = []
    for item in raw:
        if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0:
            labels.append(str(item[0]))
        else:
            labels.append(str(item))
    labels = [normalize_marker_label(lab) for lab in labels]
    return mt, labels


# -----------------------------
# Stream helpers
# -----------------------------
def find_stream_by_name(streams: List[dict], name: str) -> dict:
    for s in streams:
        if s["info"]["name"][0] == name:
            return s
    raise RuntimeError(f"Missing stream: {name}")


def estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise RuntimeError("Cannot estimate fs (bad timestamps).")
    return float(1.0 / np.median(dt))


def bandpass_ppg(x: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 8.0) -> np.ndarray:
    if sig is None:
        raise RuntimeError("scipy is required for bandpass filtering. Install: pip install scipy")
    nyq = fs / 2.0
    high = min(high_hz, nyq - 0.1)
    if high <= low_hz:
        return x
    sos = sig.butter(4, [low_hz, high], btype="bandpass", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, x)


def zscore_window(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(np.nanmean(w))
    s = float(np.nanstd(w))
    if not np.isfinite(s) or s < eps:
        return w - m
    return (w - m) / s


# -----------------------------
# Segment construction
# -----------------------------
def build_segments(marker_times: np.ndarray, marker_labels: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Uses your confirmed marker labels:
      baseline, start, longer_version_meditation, ..., eyes_open, stop

    We define:
      PRE      = [baseline, longer_version_meditation)
      MED_MAIN = [longer_version_meditation, eyes_open)
      MED_1    = first half of MED_MAIN
      MED_2    = second half of MED_MAIN
    """
    label_to_time = {lab: float(t) for lab, t in zip(marker_labels, marker_times)}

    required = ["baseline", "longer_version_meditation", "eyes_open"]
    for r in required:
        if r not in label_to_time:
            raise RuntimeError(f"Missing required marker '{r}'. Found: {sorted(label_to_time.keys())}")

    pre_start = label_to_time["baseline"]
    pre_end = label_to_time["longer_version_meditation"]

    med_start = label_to_time["longer_version_meditation"]
    med_end = label_to_time["eyes_open"]

    if not (pre_end > pre_start and med_end > med_start):
        raise RuntimeError("Bad marker ordering (check marker times).")

    mid = 0.5 * (med_start + med_end)

    return {
        "pre": (pre_start, pre_end),
        "med_1": (med_start, mid),
        "med_2": (mid, med_end),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", default="data/raw/session_01.xdf", help="Path to XDF file")
    ap.add_argument("--out_root", default="analysis_rawppg_lle/outputs", help="Output folder")
    ap.add_argument("--fs_target", type=float, default=100.0, help="(For reference) expected PPG fs")
    ap.add_argument("--window_s", type=float, default=10.0, help="Window length in seconds")
    ap.add_argument("--step_s", type=float, default=1.0, help="Step size in seconds")
    ap.add_argument("--bandpass", action="store_true", help="Apply 0.5–8 Hz bandpass before windowing")
    ap.add_argument("--low_hz", type=float, default=0.5)
    ap.add_argument("--high_hz", type=float, default=8.0)

    # Your SNR-based best-channel choices (1-based channel index)
    ap.add_argument("--best_channels", default="",
                    help="Override mapping: EmotiBit_1=1,EmotiBit_7=3 etc (channels are 1-based).")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Default mapping from YOUR printed SNR results
    best_ch_map = {
        "EmotiBit_1": 1,
        "EmotiBit_2": 1,
        "EmotiBit_3": 1,
        "EmotiBit_4": 1,
        "EmotiBit_5": 1,
        "EmotiBit_6": 1,
        "EmotiBit_7": 3,
    }

    # Optional override
    if args.best_channels.strip():
        items = [x.strip() for x in args.best_channels.split(",") if x.strip()]
        for it in items:
            k, v = it.split("=")
            best_ch_map[k.strip()] = int(v.strip())

    print(f"Loading XDF: {args.xdf}")
    streams, _ = pyxdf.load_xdf(args.xdf)

    marker_stream = find_marker_stream(streams)
    mt, labs = extract_markers(marker_stream)
    segs = build_segments(mt, labs)
    print("Segments (sec):")
    for k, (a, b) in segs.items():
        print(f"  {k:6s}: {a:.3f} -> {b:.3f}  (dur={b-a:.1f}s)")

    # Collect raw windows
    X_list = []
    meta_rows = []

    # PPG streams are named like: PPG_EmotiBit_1 ... PPG_EmotiBit_7
    subjects = sorted(best_ch_map.keys(), key=lambda s: int(s.split("_")[-1]))

    for subj in subjects:
        ppg_name = f"PPG_{subj}"
        s = find_stream_by_name(streams, ppg_name)

        data = np.asarray(s["time_series"], dtype=float)  # shape: [N, n_ch]
        t = np.asarray(s["time_stamps"], dtype=float)     # shape: [N]
        fs = estimate_fs(t)

        n_ch = data.shape[1]
        ch_1based = best_ch_map[subj]
        ch_idx = ch_1based - 1
        if ch_idx < 0 or ch_idx >= n_ch:
            raise RuntimeError(f"{ppg_name}: requested channel {ch_1based} but n_ch={n_ch}")

        x = data[:, ch_idx].astype(float)
        # basic finite cleanup
        x[~np.isfinite(x)] = np.nan
        # simple fill (avoid crashes); filtering later uses finite values
        if np.any(~np.isfinite(x)):
            med = float(np.nanmedian(x))
            x[~np.isfinite(x)] = med

        if args.bandpass:
            x = bandpass_ppg(x, fs=fs, low_hz=args.low_hz, high_hz=args.high_hz)

        FS_NOM = float(args.fs_target)  # nominal 100Hz so all subjects have same window length
        win_n = int(args.window_s * FS_NOM)
        step_n = int(args.step_s * FS_NOM)

        if win_n < 10 or step_n < 1:
            raise RuntimeError("Window/step too small after fs_target conversion.")

        print(f"\n{subj} | {ppg_name} | fs={fs:.2f} Hz | best_ch={ch_1based} | win_n={win_n} step_n={step_n}")

        for seg_name, (a, b) in segs.items():
            mask = (t >= a) & (t < b)
            idx = np.where(mask)[0]
            if len(idx) < win_n:
                print(f"  - {seg_name}: too short (samples={len(idx)}) -> skip")
                continue

            # use contiguous slice for windowing
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
            xt = x[i0:i1]
            tt = t[i0:i1]

            n_possible = 1 + (len(xt) - win_n) // step_n
            print(f"  - {seg_name}: samples={len(xt)} windows={n_possible}")

            for wi in range(n_possible):
                j0 = wi * step_n
                j1 = j0 + win_n
                w = xt[j0:j1].astype(float)

                # per-window normalize (helps LLE focus on shape, not amplitude)
                w = zscore_window(w)

                # metadata timing
                t_start = float(tt[j0])
                t_end = float(tt[j1 - 1])
                t_center = 0.5 * (t_start + t_end)
                t_rel = float(t_center - a)

                X_list.append(w)
                meta_rows.append({
                    "subject": subj,
                    "ppg_stream": ppg_name,
                    "ppg_channel_1based": ch_1based,
                    "fs": fs,
                    "segment": seg_name,
                    "t_start": t_start,
                    "t_end": t_end,
                    "t_center": t_center,
                    "t_rel": t_rel,
                    "window_s": args.window_s,
                    "step_s": args.step_s,
                    "bandpass": bool(args.bandpass),
                    "low_hz": args.low_hz if args.bandpass else np.nan,
                    "high_hz": args.high_hz if args.bandpass else np.nan,
                })

    if len(X_list) == 0:
        raise RuntimeError("No windows were extracted (check markers/segment durations).")

    lens = [w.shape[0] for w in X_list]
    uniq_lens = sorted(set(lens))
    print("\nUnique window lengths:", uniq_lens[:20], "..." if len(uniq_lens) > 20 else "")
    if len(uniq_lens) != 1:
        raise RuntimeError(f"Window length mismatch still exists: {uniq_lens}")
    
    X = np.stack(X_list, axis=0)  # [n_windows, win_n]
    meta = pd.DataFrame(meta_rows)

    out_X = out_root / "rawppg_windows_X.npy"
    out_meta = out_root / "rawppg_windows_meta.csv"

    np.save(out_X, X)
    meta.to_csv(out_meta, index=False)

    print(f"\nSaved X:    {out_X}   shape={X.shape}")
    print(f"Saved meta: {out_meta} rows={len(meta)}")
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
python3 analysis_rawppg_lle/code/01_extract_raw_windows.py \
  --xdf data/raw/session_01.xdf \
  --out_root analysis_rawppg_lle/outputs \
  --window_s 10 --step_s 1 \
  --bandpass
"""
