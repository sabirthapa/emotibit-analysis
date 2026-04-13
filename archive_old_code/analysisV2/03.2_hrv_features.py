# analysisV2/03.2_hrv_features.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def rolling_hrv(t_ibi, ibi_s, t0, t1, fs_out=1.0, window_s=60):
    """
    Compute HRV features on a uniform 1 Hz grid using a sliding window.
    window_s: 60s is common; you can also try 30s.
    """
    grid = np.arange(np.ceil(t0), np.floor(t1) + 1e-9, 1.0 / fs_out)

    rmssd = np.full(len(grid), np.nan, dtype=float)
    sdnn = np.full(len(grid), np.nan, dtype=float)
    n_beats = np.zeros(len(grid), dtype=int)

    half = window_s / 2.0

    for i, tg in enumerate(grid):
        a = tg - half
        b = tg + half
        m = (t_ibi >= a) & (t_ibi <= b)
        x = ibi_s[m]

        # need enough beats in window
        if len(x) < 5:
            continue

        # convert to ms for HRV convention
        x_ms = x * 1000.0
        sdnn[i] = np.std(x_ms, ddof=1)

        dx = np.diff(x_ms)
        rmssd[i] = np.sqrt(np.mean(dx * dx)) if len(dx) >= 2 else np.nan
        n_beats[i] = len(x)

    return pd.DataFrame({
        "time": grid,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "n_beats_in_window": n_beats
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_raw_root", default="analysis_outputs/hr_raw_v2",
                    help="Folder that contains EmotiBit_*/<segment>/IBI.csv")
    ap.add_argument("--out_root", default="analysis_outputs/hrv_v2",
                    help="Where to save HRV outputs")
    ap.add_argument("--fs_out", type=float, default=1.0)
    ap.add_argument("--window_s", type=int, default=60)
    args = ap.parse_args()

    hr_raw_root = Path(args.hr_raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for ibi_csv in hr_raw_root.glob("EmotiBit_*/**/IBI.csv"):
        eb = ibi_csv.parts[-3]
        seg = ibi_csv.parts[-2]

        df = pd.read_csv(ibi_csv)
        if not {"time", "ibi_s"}.issubset(df.columns):
            continue

        t_ibi = df["time"].to_numpy(dtype=float)
        ibi_s = df["ibi_s"].to_numpy(dtype=float)

        if len(t_ibi) < 10:
            continue

        t0, t1 = float(t_ibi.min()), float(t_ibi.max())
        hrv = rolling_hrv(t_ibi, ibi_s, t0, t1, fs_out=args.fs_out, window_s=args.window_s)

        out_dir = out_root / eb / seg
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"HRV_{int(args.fs_out)}Hz.csv"
        hrv.to_csv(out_path, index=False)

        valid_pct = 100.0 * np.mean(np.isfinite(hrv["rmssd_ms"]))
        rows.append({
            "emotibit": eb,
            "segment": seg,
            "window_s": args.window_s,
            "valid_pct": round(valid_pct, 2),
            "out": str(out_path)
        })

    summ = pd.DataFrame(rows).sort_values(["segment", "emotibit"])
    summ_path = out_root / "HRV_summary.csv"
    summ.to_csv(summ_path, index=False)

    print(f"Saved HRV under: {out_root}")
    print(f"Saved summary: {summ_path}")
    print(summ.to_string(index=False))

if __name__ == "__main__":
    main()