import os
import math
import argparse
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

# Helpers / data containers

@dataclass
class ParticipantStreams:
    serial: str
    ppg: Optional[pd.DataFrame]  # columns: PPG1,PPG2,PPG3,t
    eda: Optional[pd.DataFrame]  # columns: EDA,t
    temp: Optional[pd.DataFrame] # columns: Temperature,t


@dataclass
class SegmentTimes:
    baseline: Optional[Tuple[float, float]]
    meditation: Optional[Tuple[float, float]]
    recovery: Optional[Tuple[float, float]]


def effective_fs(t: np.ndarray) -> float:
    """Robust effective sampling rate from timestamps (ignores large gaps)."""
    if t is None or len(t) < 3:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return np.nan
    median_dt = np.median(dt)
    if median_dt <= 0 or not np.isfinite(median_dt):
        return np.nan
    return 1.0 / median_dt


def slice_by_time(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    mask = (df["t"] >= t_start) & (df["t"] <= t_end)
    return df.loc[mask].copy()


def first_valid(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


# XDF parsing

def _stream_name(s):
    return s["info"]["name"][0] if "name" in s["info"] and len(s["info"]["name"]) else ""


def _stream_source_id(s):
    # We set LSL StreamInfo(..., source_id=f"ppg_{serial}") etc.
    try:
        return s["info"]["source_id"][0]
    except Exception:
        return ""


def load_streams_grouped_by_serial(xdf_path: str) -> Tuple[Dict[str, ParticipantStreams], List[dict]]:
    """Read XDF and group PPG/EDA/TEMP by device serial (from source_id). Also return marker streams (list)."""
    streams, header = pyxdf.load_xdf(xdf_path)
    grouped: Dict[str, ParticipantStreams] = {}
    marker_streams: List[dict] = []

    for s in streams:
        name = _stream_name(s)
        sid = _stream_source_id(s)

        # Try to extract serial from source_id like "ppg_EM-V6-0000228"
        serial = None
        if "_" in sid:
            serial = sid.split("_", 1)[-1].strip()
        # If missing, fallback to name suffix like "PPG_EmotiBit_1" (less ideal)
        if serial is None or serial == "":
            # last token after underscore may be a label; we keep None to still parse data
            serial = "UNKNOWN"

        # Convert to DataFrame
        ts = np.asarray(s["time_stamps"])
        if name.startswith("PPG"):
            arr = np.asarray(s["time_series"])
            # expect shape [N, 3]
            if arr.ndim == 2 and arr.shape[1] >= 3:
                df = pd.DataFrame(arr[:, :3], columns=["PPG1", "PPG2", "PPG3"])
            elif arr.ndim == 1:
                df = pd.DataFrame({"PPG1": arr})
                df["PPG2"] = np.nan
                df["PPG3"] = np.nan
            else:
                # pad/truncate to 3
                cols = min(3, arr.shape[1] if arr.ndim == 2 else 1)
                data = np.zeros((arr.shape[0], 3))
                data[:, :cols] = arr[:, :cols] if arr.ndim == 2 else arr.reshape(-1,1)
                df = pd.DataFrame(data, columns=["PPG1", "PPG2", "PPG3"])
            df["t"] = ts

            ps = grouped.get(serial, ParticipantStreams(serial, None, None, None))
            ps.ppg = df
            grouped[serial] = ps

        elif name.startswith("EDA"):
            arr = np.asarray(s["time_series"])
            arr = arr.flatten() if arr.ndim > 1 and arr.shape[1] == 1 else arr
            df = pd.DataFrame({"EDA": arr.squeeze()})
            df["t"] = ts

            ps = grouped.get(serial, ParticipantStreams(serial, None, None, None))
            ps.eda = df
            grouped[serial] = ps

        elif name.startswith("TEMP"):
            arr = np.asarray(s["time_series"])
            arr = arr.flatten() if arr.ndim > 1 and arr.shape[1] == 1 else arr
            df = pd.DataFrame({"Temperature": arr.squeeze()})
            df["t"] = ts

            ps = grouped.get(serial, ParticipantStreams(serial, None, None, None))
            ps.temp = df
            grouped[serial] = ps

        elif "Marker" in name or "marker" in name or name.lower().startswith("marker"):
            marker_streams.append(s)

    return grouped, marker_streams


def parse_marker_segments(marker_streams: List[dict]) -> SegmentTimes:
    """
    Look across all marker streams and infer (start, end) for baseline / meditation / recovery.
    Expected marker labels (case-insensitive): baseline_start, meditation_start, recovery_start, session_end
    We will be tolerant: also accept 'baseline', 'meditation', 'recovery' as starts.
    """
    labels = []
    for s in marker_streams:
        data = s["time_series"]
        ts = s["time_stamps"]
        for v, t in zip(data, ts):
            try:
                lab = str(v[0]).strip().lower()
            except Exception:
                lab = str(v).strip().lower()
            labels.append((t, lab))

    if not labels:
        return SegmentTimes(None, None, None)

    labels.sort(key=lambda x: x[0])  # by time

    # find starts
    t_baseline = None
    t_meditation = None
    t_recovery = None
    t_end = None

    for t, lab in labels:
        if t_baseline is None and ("baseline" in lab):
            t_baseline = t
        elif t_meditation is None and ("meditation" in lab):
            t_meditation = t
        elif t_recovery is None and ("recovery" in lab):
            t_recovery = t
        elif ("end" in lab) or ("session_end" in lab) or ("stop" in lab):
            t_end = t

    # Construct windows with best effort
    # baseline: [baseline_start, meditation_start)
    # meditation: [meditation_start, recovery_start)
    # recovery: [recovery_start, end)
    baseline = None
    meditation = None
    recovery = None

    if t_baseline is not None and t_meditation is not None:
        baseline = (t_baseline, t_meditation)
    if t_meditation is not None and t_recovery is not None:
        meditation = (t_meditation, t_recovery)
    if t_recovery is not None:
        recovery = (t_recovery, first_valid(t_end, t_recovery + 1e9))  # open-ended if no end

    return SegmentTimes(baseline, meditation, recovery)


# Metrics

def compute_hr_from_ppg(ppg_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Returns (hr_series_df with columns t, HR, fs, and processed dict), effective_fs.
    Uses PPG1 channel by default; if flat/noisy, tries PPG2 then PPG3.
    """
    if ppg_df is None or ppg_df.empty:
        return pd.DataFrame(columns=["t","HR"]), np.nan

    t = ppg_df["t"].to_numpy()
    fs_eff = effective_fs(t)
    if not np.isfinite(fs_eff) or fs_eff <= 0:
        # fallback to declared 100 Hz if your LSL defined it that way
        fs_eff = 100.0

    for ch in ["PPG1", "PPG2", "PPG3"]:
        if ch not in ppg_df.columns:
            continue
        sig = ppg_df[ch].to_numpy().astype(float)
        if np.all(np.isnan(sig)) or len(sig) < 50:
            continue
        try:
            processed = nk.ppg_process(sig, sampling_rate=fs_eff)
            hr = processed[0]["PPG_Rate"]  # bpm
            # neurokit returns equally spaced; we rebuild time axis from original t using same length
            # Align by linear interpolation to original timestamps length
            # safest: use same length as signal (nk outputs len(sig))
            t_hr = np.linspace(t[0], t[-1], num=len(hr))
            hr_df = pd.DataFrame({"t": t_hr, "HR": hr})
            return hr_df, fs_eff
        except Exception:
            continue

    # if all failed
    return pd.DataFrame(columns=["t","HR"]), fs_eff


def summarize_segment(hr_df: pd.DataFrame, seg: Optional[Tuple[float,float]]) -> Dict[str, float]:
    if seg is None or hr_df.empty:
        return {"avg": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    st, en = seg
    sub = hr_df[(hr_df["t"] >= st) & (hr_df["t"] <= en)]
    if sub.empty:
        return {"avg": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    return {
        "avg": float(np.nanmean(sub["HR"])),
        "min": float(np.nanmin(sub["HR"])),
        "max": float(np.nanmax(sub["HR"])),
        "n": int(sub.shape[0]),
    }


def summarize_scalar(df: Optional[pd.DataFrame], col: str, seg: Optional[Tuple[float,float]]) -> float:
    if df is None or df.empty or seg is None:
        return np.nan
    st, en = seg
    sub = df[(df["t"] >= st) & (df["t"] <= en)]
    if sub.empty or col not in sub.columns:
        return np.nan
    return float(np.nanmean(sub[col]))


# Reporting (PDF + plots)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def plot_ppg_and_hr(serial: str, ppg_df: Optional[pd.DataFrame], hr_df: pd.DataFrame, outdir: str) -> Optional[str]:
    if ppg_df is None or ppg_df.empty or hr_df is None or hr_df.empty:
        return None
    plt.figure(figsize=(8,4))
    # Downsample PPG for plotting
    psub = ppg_df.iloc[::max(1, len(ppg_df)//500)].copy()
    plt.plot(psub["t"]-psub["t"].iloc[0], psub["PPG1"], label="PPG1", alpha=0.7)
    plt.plot(hr_df["t"]-hr_df["t"].iloc[0], hr_df["HR"]/100.0, label="HR/100 (scaled)", alpha=0.9)  # scale to overlay
    plt.xlabel("Time (s, relative)")
    plt.ylabel("Amplitude / (bpm/100)")
    plt.title(f"PPG & HR Trend — {serial}")
    plt.legend()
    ensure_dir(os.path.join(outdir, "PPG_QC_plots"))
    png_path = os.path.join(outdir, "PPG_QC_plots", f"{serial}_ppg_hr.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return png_path


def make_pdf(serial: str,
             session_name: str,
             outdir: str,
             hr_df: pd.DataFrame,
             fs_ppg: float,
             segs: SegmentTimes,
             eda_means: Dict[str, float],
             temp_means: Dict[str, float],
             indiv_metrics: Dict[str, Dict[str, float]],
             class_metrics: Dict[str, Dict[str, float]],
             plot_path: Optional[str]) -> str:

    pdf_path = os.path.join(outdir, f"report_{serial}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter

    # Title
    y = h - 1.0*inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1.0*inch, y, f"Heart Meditation Report — {serial}")
    y -= 0.3*inch
    c.setFont("Helvetica", 11)
    c.drawString(1.0*inch, y, f"Session: {session_name}")
    y -= 0.2*inch
    c.drawString(1.0*inch, y, f"PPG effective sampling rate: {fs_ppg:.2f} Hz")
    y -= 0.35*inch

    # If we have a plot, add it
    if plot_path and os.path.exists(plot_path):
        c.drawImage(plot_path, 1.0*inch, y-3.2*inch, width=5.5*inch, height=3.0*inch, preserveAspectRatio=True)
        y -= 3.4*inch

    # Segment metrics
    def fmt(seg, m):
        if seg not in indiv_metrics: return "—"
        avg = indiv_metrics[seg].get("avg", np.nan)
        mn  = indiv_metrics[seg].get("min", np.nan)
        mx  = indiv_metrics[seg].get("max", np.nan)
        if not np.isfinite(avg): return "—"
        return f"Avg {avg:.1f} bpm (Min {mn:.1f}, Max {mx:.1f})"

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0*inch, y, "Heart Rate Summary")
    y -= 0.2*inch
    c.setFont("Helvetica", 11)
    c.drawString(1.0*inch, y, f"Baseline:   {fmt('baseline', indiv_metrics)}")
    y -= 0.2*inch
    c.drawString(1.0*inch, y, f"Meditation: {fmt('meditation', indiv_metrics)}")
    y -= 0.2*inch
    c.drawString(1.0*inch, y, f"Recovery:   {fmt('recovery', indiv_metrics)}")
    y -= 0.3*inch

    # Changes
    def pct(a, b):
        if not (np.isfinite(a) and np.isfinite(b)) or a == 0:
            return np.nan
        return 100.0 * (b - a)/a

    base_avg = indiv_metrics.get("baseline", {}).get("avg", np.nan)
    med_avg  = indiv_metrics.get("meditation", {}).get("avg", np.nan)
    rec_avg  = indiv_metrics.get("recovery", {}).get("avg", np.nan)
    p_med  = pct(base_avg, med_avg)
    p_rec  = pct(base_avg, rec_avg)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0*inch, y, "Changes (↓ lower HR implies relaxation)")
    y -= 0.2*inch
    c.setFont("Helvetica", 11)
    c.drawString(1.0*inch, y, f"Δ Baseline→Meditation: {('+' if p_med>0 else '')}{p_med:.1f}%")
    y -= 0.2*inch
    c.drawString(1.0*inch, y, f"Δ Baseline→Recovery:   {('+' if p_rec>0 else '')}{p_rec:.1f}%")
    y -= 0.3*inch

    # Class comparison
    def fmt_cls(seg):
        m = class_metrics.get(seg, {})
        if not m: return "—"
        avg = m.get("avg", np.nan)
        if not np.isfinite(avg): return "—"
        return f"{avg:.1f} bpm"

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0*inch, y, "Class Averages")
    y -= 0.2*inch
    c.setFont("Helvetica", 11)
    c.drawString(1.0*inch, y, f"Baseline:   {fmt_cls('baseline')}")
    y -= 0.2*inch
    c.drawString(1.0*inch, y, f"Meditation: {fmt_cls('meditation')}")
    y -= 0.2*inch
    c.drawString(1.0*inch, y, f"Recovery:   {fmt_cls('recovery')}")
    y -= 0.3*inch

    # EDA / Temp
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0*inch, y, "Other Signals")
    y -= 0.2*inch
    c.setFont("Helvetica", 11)
    c.drawString(1.0*inch, y, f"EDA mean (meditation): {eda_means.get('meditation', np.nan):.4f} (a.u.)")
    y -= 0.2*inch
    c.drawString(1.0*inch, y, f"Temp mean (meditation): {temp_means.get('meditation', np.nan):.2f} °C")
    y -= 0.4*inch

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(1.0*inch, y, "Notes: HR derived from PPG; Temp in °C; EDA is relative conductance.")
    c.showPage()
    c.save()
    return pdf_path


# Main pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", required=True, help="Path to the session .xdf")
    ap.add_argument("--out", required=True, help="Output folder for PDFs/plots")
    ap.add_argument("--session", default="Meditation Session", help="Session name printed in report")
    args = ap.parse_args()

    outdir = ensure_dir(args.out)

    grouped, marker_streams = load_streams_grouped_by_serial(args.xdf)
    segs = parse_marker_segments(marker_streams)

    # per participant metrics
    per_participant = []

    # First pass: compute individual HR & segment stats
    indiv_results = {}
    for serial, ps in grouped.items():
        # Compute HR
        hr_df, fs_ppg = compute_hr_from_ppg(ps.ppg)

        # Segment summaries
        m_bas = summarize_segment(hr_df, segs.baseline)
        m_med = summarize_segment(hr_df, segs.meditation)
        m_rec = summarize_segment(hr_df, segs.recovery)

        # EDA/Temp means (for meditation, but you can extend similarly)
        eda_means = {
            "baseline":   summarize_scalar(ps.eda, "EDA", segs.baseline),
            "meditation": summarize_scalar(ps.eda, "EDA", segs.meditation),
            "recovery":   summarize_scalar(ps.eda, "EDA", segs.recovery),
        }
        temp_means = {
            "baseline":   summarize_scalar(ps.temp, "Temperature", segs.baseline),
            "meditation": summarize_scalar(ps.temp, "Temperature", segs.meditation),
            "recovery":   summarize_scalar(ps.temp, "Temperature", segs.recovery),
        }

        indiv_results[serial] = {
            "hr_df": hr_df,
            "fs_ppg": fs_ppg,
            "eda_means": eda_means,
            "temp_means": temp_means,
            "segments": {
                "baseline": m_bas,
                "meditation": m_med,
                "recovery": m_rec
            },
        }

        # store for class averages (use averages only)
        per_participant.append({
            "serial": serial,
            "baseline_avg": m_bas["avg"],
            "meditation_avg": m_med["avg"],
            "recovery_avg": m_rec["avg"],
        })

    # Class averages
    cls_df = pd.DataFrame(per_participant)
    class_metrics = {}
    for seg in ["baseline", "meditation", "recovery"]:
        col = f"{seg}_avg"
        if col in cls_df.columns:
            class_metrics[seg] = {"avg": float(np.nanmean(cls_df[col]))}
        else:
            class_metrics[seg] = {"avg": np.nan}

    # Generate PDFs with plots
    for serial, r in indiv_results.items():
        hr_df  = r["hr_df"]
        fs_ppg = r["fs_ppg"]
        eda_means = r["eda_means"]
        temp_means = r["temp_means"]
        seg_metrics = r["segments"]

        png_path = plot_ppg_and_hr(serial, grouped[serial].ppg, hr_df, outdir)
        pdf_path = make_pdf(
            serial=serial,
            session_name=args.session,
            outdir=outdir,
            hr_df=hr_df,
            fs_ppg=fs_ppg,
            segs=segs,
            eda_means=eda_means,
            temp_means=temp_means,
            indiv_metrics={
                "baseline": seg_metrics["baseline"],
                "meditation": seg_metrics["meditation"],
                "recovery": seg_metrics["recovery"],
            },
            class_metrics=class_metrics,
            plot_path=png_path,
        )
        print(f"✓ Wrote {pdf_path}")

    print("\nAll done. Reports in:", outdir)


if __name__ == "__main__":
    main()