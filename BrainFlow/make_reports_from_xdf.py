import os, math, argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from reportlab.lib import colors
import re

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

# data Structures 

UB_BLUE = colors.HexColor("#005BBB")

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

def first_valid(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

# XDF Parsing 

def _stream_name(s): 
    return s["info"]["name"][0] if "name" in s["info"] else ""

def _stream_source_id(s):
    try:
        return s["info"]["source_id"][0]
    except Exception:
        return ""

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

# Start/Stop-only marker parser

def _is_start(lbl: str) -> bool:
    lbl = lbl.lower()
    return bool(re.search(r"\b(start|begin|meditation start)\b", lbl))

def _is_stop(lbl: str) -> bool:
    lbl = lbl.lower()
    return bool(re.search(r"\b(stop|end|finish|meditation stop)\b", lbl))

def parse_start_stop(marker_streams: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (t_start, t_stop) using only 'start'/'stop' style markers.
    Pick earliest start; pick first stop after that start. Ignore all other labels.
    """
    events = []
    for s in marker_streams:
        for v, t in zip(s["time_series"], s["time_stamps"]):
            try:
                lab = str(v[0])
            except Exception:
                lab = str(v)
            events.append((t, lab))
    if not events:
        return None, None

    events.sort(key=lambda x: x[0])
    starts = [t for t, lab in events if _is_start(lab)]
    if not starts:
        return None, None
    t_start = starts[0]

    stops_after = [t for t, lab in events if _is_stop(lab) and t > t_start]
    t_stop = stops_after[0] if stops_after else None
    return t_start, t_stop

# ---------------- Metrics ----------------

def compute_hr_from_ppg(ppg_df: pd.DataFrame):
    if ppg_df is None or ppg_df.empty:
        return pd.DataFrame(columns=["t","HR"]), np.nan

    t = ppg_df["t"].to_numpy()
    fs_eff = effective_fs(t) or 100.0

    for ch in ["PPG1","PPG2","PPG3"]:
        sig = ppg_df[ch].to_numpy(dtype=float)
        if np.all(np.isnan(sig)) or len(sig) < 50:
            continue
        try:
            proc = nk.ppg_process(sig, sampling_rate=fs_eff)
            hr = proc[0]["PPG_Rate"]

            window_size = int(fs_eff * 5)
            hr_smooth = pd.Series(hr).rolling(window=window_size, center=True, min_periods=1).mean()

            t_hr = np.linspace(t[0], t[-1], len(hr_smooth))
            return pd.DataFrame({"t": t_hr, "HR": hr_smooth}), fs_eff
        except Exception:
            continue
    return pd.DataFrame(columns=["t","HR"]), fs_eff

def summarize_segment(hr_df, seg):
    if seg is None or hr_df.empty:
        return np.nan
    st, en = seg
    sub = hr_df[(hr_df["t"] >= st) & (hr_df["t"] <= en)]
    return np.nanmean(sub["HR"]) if not sub.empty else np.nan

# Plot & PDF

def plot_hr_trend(serial, hr_df, segs, outdir):
    if hr_df.empty:
        return None

    plt.figure(figsize=(7.0, 3.2))
    plt.plot(hr_df["t"] - hr_df["t"].iloc[0], hr_df["HR"], color="#2a9d8f", lw=1.5)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heart Rate (bpm)")

    plt.title(f"Heart Rate Trend — {serial}", fontsize=11, pad=2)
    plt.ylim(40, 120)
    plt.tight_layout(pad=0.2)

    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    path = os.path.join(outdir, "plots", f"{serial}_hr.png")
    plt.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    return path

def make_pdf(serial, hr_avgs, class_avgs, temp_mean, eda_mean, fs, plot_path, outdir, user_name, user_email):
    pdf_path = os.path.join(outdir, f"report_{serial}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter

    # UB logo (keeps natural aspect ratio)
    logo_path = os.path.join(os.path.dirname(__file__), "ub_logo.png")
    if os.path.exists(logo_path):
        try:
            from PIL import Image
            logo_img = Image.open(logo_path)
            aspect_ratio = logo_img.height / logo_img.width
            logo_width = 2.0 * inch
            logo_height = logo_width * aspect_ratio
            c.drawImage(
                logo_path,
                (w - logo_width) / 2,
                h - (1.0 * inch + logo_height / 2),
                width=logo_width,
                height=logo_height,
                mask="auto"
            )
        except Exception:
            # fallback if Pillow not available
            c.drawImage(logo_path, (w - 2.0*inch)/2, h - 1.6*inch, width=2.0*inch, height=0.6*inch, mask="auto")

    left_col = 1.2 * inch
    right_col = 2.9 * inch

    # title
    y = h - 1.6 * inch
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(UB_BLUE)
    c.drawString(left_col, y, "Meditation Report")
    y -= 0.4 * inch

    # participant info
    info = [
        ("Participant:", user_name or "—"),
        ("Email:", user_email or "—"),
        ("Device ID:", serial),
        ("Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    for label, value in info:
        c.setFont("Helvetica-Bold", 11); c.setFillColor(UB_BLUE); c.drawString(left_col, y, label)
        c.setFont("Helvetica", 11); c.setFillColor(colors.black); c.drawString(right_col, y, value)
        y -= 0.25 * inch

    # divider
    y -= 0.05 * inch
    c.setStrokeColor(colors.lightgrey)
    c.line(left_col, y, w - left_col, y)
    y -= 0.4 * inch

    # study desc
    c.setFont("Helvetica", 11); c.setFillColor(colors.black)
    c.drawString(left_col, y, "Heart rate measured during your meditation session.")
    y -= 0.25 * inch
    c.drawString(left_col, y, f"Sampling rate: {fs:.2f} Hz")
    y -= 0.45 * inch

    # heart rate plot
    if plot_path and os.path.exists(plot_path):
        c.drawImage(plot_path, left_col, y - 2.8 * inch, width=5.7 * inch, height=2.8 * inch)
        y -= 3.0 * inch

    # heart rates
    y -= 0.25 * inch
    c.setFont("Helvetica-Bold", 12); c.setFillColor(UB_BLUE)
    c.drawString(left_col, y, "Your Average Heart Rates")
    y -= 0.25 * inch
    c.setFont("Helvetica", 11); c.setFillColor(colors.black)
    for k in ["Before", "After"]:
        c.drawString(left_col, y, f"{k:8s} — Your Avg: {hr_avgs[k]:5.1f} bpm   Group Avg: {class_avgs[k]:5.1f} bpm")
        y -= 0.25 * inch
    y -= 0.35 * inch

    # interpretation
    delta = hr_avgs["After"] - hr_avgs["Before"]
    c.setFont("Helvetica-Bold", 12); c.setFillColor(UB_BLUE)
    c.drawString(left_col, y, "Interpretation")
    y -= 0.25 * inch
    c.setFont("Helvetica", 11); c.setFillColor(colors.black)
    if np.isfinite(delta):
        if delta < 0:
            c.drawString(left_col, y, f"Your heart rate decreased by {abs(delta):.1f} bpm.")
        else:
            c.drawString(left_col, y, f"Your heart rate increased by {delta:.1f} bpm.")
    else:
        c.drawString(left_col, y, "Not enough data to compute a change.")
    y -= 0.5 * inch

    # other signals
    c.setFont("Helvetica-Bold", 12); c.setFillColor(UB_BLUE)
    c.drawString(left_col, y, "Other Signals")
    y -= 0.25 * inch
    c.setFont("Helvetica", 11); c.setFillColor(colors.black)
    if np.isfinite(temp_mean):
        c.drawString(left_col, y, f"Average Skin Temperature: {temp_mean:.2f} °C")
    else:
        c.drawString(left_col, y, "Average Skin Temperature: —")

    # footer
    c.setFont("Helvetica-Oblique", 9); c.setFillColor(colors.grey)
    c.drawString(left_col, 0.55 * inch, "Note: Lower heart rates after meditation usually indicate greater relaxation.")

    c.save()
    return pdf_path

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default="participants.csv",
                    help="CSV file mapping serial numbers to participant info (name,email)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # participant mapping
    if os.path.exists(args.map):
        mapping_df = pd.read_csv(args.map)
        name_lookup = dict(zip(mapping_df.serial, mapping_df.name))
        email_lookup = dict(zip(mapping_df.serial, mapping_df.email))
    else:
        print(f"⚠️ Warning: mapping file '{args.map}' not found — using generic names.")
        name_lookup, email_lookup = {}, {}

    # load
    grouped, marker_streams = load_streams_grouped_by_serial(args.xdf)
    t_start, t_stop = parse_start_stop(marker_streams)  # <-- START/STOP only

    all_avgs = []
    indiv = {}

    for serial, ps in grouped.items():
        hr_df, fs = compute_hr_from_ppg(ps.ppg)

        # derive per-participant segments by clamping to that participant's HR range
        if hr_df.empty:
            segs_this = SegmentTimes(None, None, None)
        else:
            t0 = float(hr_df["t"].iloc[0])
            t1 = float(hr_df["t"].iloc[-1])

            before = (t0, t_start) if (t_start is not None and t_start > t0) else None

            during_start = t_start if (t_start is not None and t_start > t0) else t0
            if t_stop is not None and t_stop > during_start:
                during_end = min(t_stop, t1)
            else:
                during_end = t1
            during = (during_start, during_end) if during_end > during_start else None

            after = (t_stop, t1) if (t_stop is not None and t_stop < t1) else None
            segs_this = SegmentTimes(before, during, after)

        hr_avgs = {
            "Before": summarize_segment(hr_df, segs_this.before),
            "During": summarize_segment(hr_df, segs_this.during),
            "After":  summarize_segment(hr_df, segs_this.after)
        }

        indiv[serial] = {
            "hr_df": hr_df,
            "fs": fs,
            "hr_avgs": hr_avgs,
            "segs": segs_this,
            "eda_mean": np.nanmean(ps.eda["EDA"]) if ps.eda is not None else np.nan,
            "temp_mean": np.nanmean(ps.temp["Temperature"]) if ps.temp is not None else np.nan
        }
        all_avgs.append(hr_avgs)

    # class averages
    class_avgs = {}
    for k in ["Before","During","After"]:
        vals = [x[k] for x in all_avgs if not math.isnan(x[k])]
        class_avgs[k] = np.nanmean(vals) if vals else np.nan

    # generate reports
    for serial, r in indiv.items():
        plot_path = plot_hr_trend(serial, r["hr_df"], r["segs"], args.out)
        user_name = name_lookup.get(serial, "")
        user_email = email_lookup.get(serial, "")
        make_pdf(serial, r["hr_avgs"], class_avgs, r["temp_mean"], r["eda_mean"],
                 r["fs"], plot_path, args.out, user_name, user_email)
        print(f"✅ Saved report for {serial}")

    print("\n🎉 All reports generated successfully.")

if __name__ == "__main__":
    main()