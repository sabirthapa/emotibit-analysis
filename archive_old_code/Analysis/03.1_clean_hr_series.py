import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def clean_hr(
    df,
    hr_min=40,
    hr_max=180,
    max_jump=15,
    max_gap_s=5,
    smooth_s=5,
    fs=1.0,
):
    """
    df: columns [time, hr_bpm, is_valid] at uniform fs (1 Hz recommended)
    Produces: hr_bpm_clean, is_valid (based on cleaned series)
    """
    x = df["hr_bpm"].to_numpy(dtype=float)

    # Range filter
    x[(x < hr_min) | (x > hr_max)] = np.nan

    # Jump filter (mark the point AFTER a big jump as bad)
    dx = np.abs(np.diff(x))
    bad = np.zeros_like(x, dtype=bool)
    bad[1:] = dx > max_jump
    x[bad] = np.nan

    # Fill short gaps (<= max_gap_s) using linear interpolation
    t = df["time"].to_numpy(dtype=float)
    isn = np.isnan(x)

    if np.any(~isn):
        x_interp = x.copy()
        x_interp[isn] = np.interp(t[isn], t[~isn], x[~isn])

        idx = np.where(isn)[0]
        if len(idx) > 0:
            runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for r in runs:
                gap_len = len(r) / fs
                if gap_len <= max_gap_s:
                    x[r] = x_interp[r]
                # else: keep NaN (long gap)

    # Smooth (moving average) ignoring NaNs
    win = int(smooth_s * fs)
    if win >= 2:
        x2 = x.copy()
        half = win // 2
        for i in range(len(x)):
            a = max(0, i - half)
            b = min(len(x), i + half + 1)
            w = x[a:b]
            x2[i] = np.nan if np.all(np.isnan(w)) else np.nanmean(w)
        x = x2

    out = df.copy()
    out["hr_bpm_clean"] = x
    out["is_valid"] = ~np.isnan(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_root", default="analysis_outputs/hr_raw", help="Root that contains HR_1Hz_raw.csv files")
    ap.add_argument("--out_root", default="analysis_outputs/hr_clean", help="Where to save cleaned HR files")
    ap.add_argument("--fs", type=float, default=1.0)

    ap.add_argument("--hr_min", type=float, default=40.0)
    ap.add_argument("--hr_max", type=float, default=180.0)
    ap.add_argument("--max_jump", type=float, default=15.0)
    ap.add_argument("--max_gap_s", type=float, default=5.0)
    ap.add_argument("--smooth_s", type=float, default=5.0)

    args = ap.parse_args()

    hr_root = Path(args.hr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    # UPDATED: match your current filename
    for hr_csv in hr_root.glob("EmotiBit_*/**/HR_1Hz_raw.csv"):
        eb = hr_csv.parts[-3]   # EmotiBit_X
        seg = hr_csv.parts[-2]  # segment folder (baseline_warmup / meditation_all)
        df = pd.read_csv(hr_csv)

        cleaned = clean_hr(
            df,
            hr_min=args.hr_min,
            hr_max=args.hr_max,
            max_jump=args.max_jump,
            max_gap_s=args.max_gap_s,
            smooth_s=args.smooth_s,
            fs=args.fs,
        )

        out_dir = out_root / eb / seg
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "HR_1Hz_clean.csv"
        cleaned.to_csv(out_path, index=False)

        valid_pct = 100.0 * cleaned["is_valid"].mean()
        rows.append(
            {
                "emotibit": eb,
                "segment": seg,
                "valid_pct": round(valid_pct, 2),
                "out": str(out_path),
            }
        )

    summ = pd.DataFrame(rows).sort_values(["segment", "emotibit"])
    summ_path = out_root / "HR_clean_summary.csv"
    summ.to_csv(summ_path, index=False)

    print(f"Saved cleaned HR under: {out_root}")
    print(f"Saved summary: {summ_path}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()

"""
python3 analysis/03.1_clean_hr_series.py \
  --hr_root analysis_outputs/hr_raw \
  --out_root analysis_outputs/hr_clean \
  --max_jump 15 \
  --max_gap_s 5 \
  --smooth_s 5
"""