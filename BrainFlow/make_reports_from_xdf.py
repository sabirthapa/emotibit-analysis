import os, math, argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyxdf
import neurokit2 as nk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Helpers / Data Structures 

@dataclass
class ParticipantStreams:
    serial: str
    ppg: Optional[pd.DataFrame]
    eda: Optional[pd.DataFrame]
    temp: Optional[pd.DataFrame]

@dataclass
class SegmentTimes:
    before: Optional[Tuple[float, float]]
    during: Optional[Tuple[float, float]]
    after: Optional[Tuple[float, float]]

def effective_fs(t: np.ndarray) -> float:
    if t is None or len(t) < 3:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    return 1.0 / np.median(dt)

def slice_by_time(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    mask = (df["t"] >= t_start) & (df["t"] <= t_end)
    return df.loc[mask].copy()

def first_valid(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

# XDF Parsing 

def _stream_name(s): return s["info"]["name"][0] if "name" in s["info"] else ""

def _stream_source_id(s):
    try: return s["info"]["source_id"][0]
    except Exception: return ""

def load_streams_grouped_by_serial(xdf_path: str):
    streams, header = pyxdf.load_xdf(xdf_path)
    grouped: Dict[str, ParticipantStreams] = {}
    marker_streams: List[dict] = []

    for s in streams:
        name = _stream_name(s)
        sid = _stream_source_id(s)
        serial = sid.split("_")[-1] if "_" in sid else "UNKNOWN"

        ts = np.asarray(s["time_stamps"])

        if name.startswith("PPG"):
            arr = np.asarray(s["time_series"])
            df = pd.DataFrame(arr[:, :3], columns=["PPG1","PPG2","PPG3"])
            df["t"] = ts
            ps = grouped.get(serial, ParticipantStreams(serial,None,None,None))
            ps.ppg = df; grouped[serial] = ps

        elif name.startswith("EDA"):
            arr = np.asarray(s["time_series"]).flatten()
            df = pd.DataFrame({"EDA": arr, "t": ts})
            ps = grouped.get(serial, ParticipantStreams(serial,None,None,None))
            ps.eda = df; grouped[serial] = ps

        elif name.startswith("TEMP"):
            arr = np.asarray(s["time_series"]).flatten()
            df = pd.DataFrame({"Temperature": arr, "t": ts})
            ps = grouped.get(serial, ParticipantStreams(serial,None,None,None))
            ps.temp = df; grouped[serial] = ps

        elif "marker" in name.lower():
            marker_streams.append(s)
    return grouped, marker_streams

def parse_marker_segments(marker_streams: List[dict]) -> SegmentTimes:
    labels = []
    for s in marker_streams:
        for val, t in zip(s["time_series"], s["time_stamps"]):
            lab = str(val[0]).lower() if isinstance(val, list) else str(val).lower()
            labels.append((t, lab))
    labels.sort(key=lambda x: x[0])

    t_before, t_during, t_after, t_end = None, None, None, None
    for t, lab in labels:
        if "base" in lab and t_before is None: t_before = t
        elif "medit" in lab and t_during is None: t_during = t
        elif "recov" in lab and t_after is None: t_after = t
        elif "end" in lab or "stop" in lab: t_end = t

    before = (t_before, t_during) if t_before and t_during else None
    during = (t_during, t_after) if t_during and t_after else None
    after = (t_after, first_valid(t_end, t_after + 60)) if t_after else None
    return SegmentTimes(before, during, after)

# Metrics 

def compute_hr_from_ppg(ppg_df: pd.DataFrame):
    if ppg_df is None or ppg_df.empty:
        return pd.DataFrame(columns=["t","HR"]), np.nan

    t = ppg_df["t"].to_numpy()
    fs_eff = effective_fs(t) or 100.0

    for ch in ["PPG1","PPG2","PPG3"]:
        sig = ppg_df[ch].to_numpy(dtype=float)
        if np.all(np.isnan(sig)) or len(sig) < 50: continue
        try:
            proc = nk.ppg_process(sig, sampling_rate=fs_eff)
            hr = proc[0]["PPG_Rate"]
            
            # ADD SMOOTHING HERE - moving average
            window_size = int(fs_eff * 5)  # 5-second window
            hr_smooth = pd.Series(hr).rolling(window=window_size, center=True, min_periods=1).mean()
            
            t_hr = np.linspace(t[0], t[-1], len(hr_smooth))
            return pd.DataFrame({"t": t_hr, "HR": hr_smooth}), fs_eff
        except Exception:
            continue
    return pd.DataFrame(columns=["t","HR"]), fs_eff

def summarize_segment(hr_df, seg):
    if seg is None or hr_df.empty: return np.nan
    st, en = seg
    sub = hr_df[(hr_df["t"] >= st) & (hr_df["t"] <= en)]
    return np.nanmean(sub["HR"]) if not sub.empty else np.nan

# Plot and PDF generation

def plot_hr_trend(serial, hr_df, segs, outdir):
    if hr_df.empty: return None
    plt.figure(figsize=(7.5,4))
    plt.plot(hr_df["t"]-hr_df["t"].iloc[0], hr_df["HR"], color="#2a9d8f", lw=1.5)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title(f"Heart Rate Trend — {serial}")
    
    plt.ylim(40, 120)  # or dynamically: plt.ylim(hr_df["HR"].quantile(0.01), hr_df["HR"].quantile(0.99))
    
    for label,(a,b) in {"Before":segs.before, "During":segs.during, "After":segs.after}.items():
        if a and b:
            plt.axvline(a-hr_df["t"].iloc[0], ls="--", alpha=0.5)
            plt.text(a-hr_df["t"].iloc[0]+2, plt.ylim()[1]*0.95, label, fontsize=9, alpha=0.7)
    os.makedirs(os.path.join(outdir,"plots"), exist_ok=True)
    path = os.path.join(outdir,"plots",f"{serial}_hr.png")
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
    return path

def make_pdf(serial, hr_avgs, class_avgs, temp_mean, eda_mean, fs, plot_path, outdir):
    pdf_path = os.path.join(outdir, f"report_{serial}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter

    y = h - 1.0*inch
    c.setFont("Helvetica-Bold",16)
    c.drawString(1.0*inch, y, f"Meditation Report — {serial}")
    y -= 0.3*inch
    c.setFont("Helvetica",11)
    c.drawString(1.0*inch, y, f"Heart rate measured during your meditation session.")
    y -= 0.25*inch
    c.drawString(1.0*inch, y, f"Sampling rate: {fs:.2f} Hz")
    y -= 0.4*inch

    if plot_path and os.path.exists(plot_path):
        c.drawImage(plot_path, 1.0*inch, y-3.0*inch, width=5.5*inch, height=3.0*inch)
        y -= 3.3*inch

    c.setFont("Helvetica-Bold",12)
    c.drawString(1.0*inch, y, "Your Average Heart Rates")
    y -= 0.2*inch; c.setFont("Helvetica",11)
    for k in ["Before","During","After"]:
        c.drawString(1.0*inch, y, f"{k:8s} — You: {hr_avgs[k]:5.1f} bpm   Class Avg: {class_avgs[k]:5.1f} bpm")
        y -= 0.2*inch
    y -= 0.3*inch

    delta = hr_avgs["After"] - hr_avgs["Before"]
    c.setFont("Helvetica-Bold",12)
    c.drawString(1.0*inch, y, "Interpretation")
    y -= 0.2*inch; c.setFont("Helvetica",11)
    if delta < 0:
        c.drawString(1.0*inch, y, f"Your heart rate decreased by {abs(delta):.1f} bpm.")
    else:
        c.drawString(1.0*inch, y, f"Your heart rate increased by {delta:.1f} bpm.")
    y -= 0.4*inch

    c.setFont("Helvetica-Bold",12)
    c.drawString(1.0*inch, y, "Other Signals")
    y -= 0.2*inch; c.setFont("Helvetica",11)
    c.drawString(1.0*inch, y, f"Average Skin Temperature: {temp_mean:.2f} °C")

    c.setFont("Helvetica-Oblique",9)
    c.drawString(1.0*inch, 0.8*inch, "Note: Lower heart rates after meditation usually indicate greater relaxation.")
    c.save()
    return pdf_path

# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grouped, marker_streams = load_streams_grouped_by_serial(args.xdf)
    segs = parse_marker_segments(marker_streams)

    all_avgs = []
    indiv = {}
    for serial, ps in grouped.items():
        hr_df, fs = compute_hr_from_ppg(ps.ppg)
        hr_avgs = {
            "Before": summarize_segment(hr_df, segs.before),
            "During": summarize_segment(hr_df, segs.during),
            "After": summarize_segment(hr_df, segs.after)
        }
        indiv[serial] = {"hr_df": hr_df, "fs": fs, "hr_avgs": hr_avgs,
                         "eda_mean": np.nanmean(ps.eda["EDA"]) if ps.eda is not None else np.nan,
                         "temp_mean": np.nanmean(ps.temp["Temperature"]) if ps.temp is not None else np.nan}
        all_avgs.append(hr_avgs)

    # compute class averages
    class_avgs = {}
    for k in ["Before","During","After"]:
        vals = [x[k] for x in all_avgs if not math.isnan(x[k])]
        class_avgs[k] = np.nanmean(vals) if vals else np.nan

    # generate reports
    for serial, r in indiv.items():
        plot_path = plot_hr_trend(serial, r["hr_df"], segs, args.out)
        make_pdf(serial, r["hr_avgs"], class_avgs, r["temp_mean"], r["eda_mean"],
                 r["fs"], plot_path, args.out)
        print(f"✅ Saved report for {serial}")

    print("\n🎉 All reports generated successfully.")

if __name__ == "__main__":
    main()