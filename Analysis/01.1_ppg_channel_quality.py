"""
PPG Channel Selection via SNR — For Heart Rate Analysis
Justification: Charlton et al. (2025) and Elgendi (2016)

Method:
  - Bandpass 0.5–8 Hz to remove baseline wander and HF noise
  - SNR = power in cardiac band (0.5–4 Hz) / power in noise band (4–8 Hz)
  - Computed across 10s windows, reported as median ± IQR
  - Higher SNR → cleaner peak detection → more accurate HR estimation

Usage:
    python snr_channel_selection.py --xdf path/to/file.xdf --out analysis_outputs/snr --show
"""

import argparse
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pyxdf
from scipy import signal as sig
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

WINDOW_SEC = 10  # 10-second windows per Elgendi (2016)


def find_stream_by_name(streams: List[dict], name: str) -> Optional[dict]:
    for s in streams:
        if s["info"]["name"][0] == name:
            return s
    return None


def find_marker_stream(streams: List[dict]) -> Optional[dict]:
    for s in streams:
        name = str(s["info"]["name"][0]).lower()
        stype = str(s["info"]["type"][0]).lower()
        if "marker" in stype or "marker" in name:
            return s
    return None


def get_emotibit_indices(streams: List[dict]) -> List[int]:
    idxs = set()
    pattern = re.compile(r"(PPG|EDA|TEMP)_EmotiBit_(\d+)$")
    for s in streams:
        name = s["info"]["name"][0]
        m = pattern.match(name)
        if m:
            idxs.add(int(m.group(2)))
    return sorted(list(idxs))


def normalize_marker_label(label: str) -> str:
    lab = str(label).strip().lower()
    lab = re.sub(r"\s+", " ", lab)
    lab = lab.replace(" ", "_")
    return lab


def extract_markers(marker_stream: dict) -> Tuple[np.ndarray, List[str]]:
    mt = np.asarray(marker_stream["time_stamps"], dtype=float)
    raw = marker_stream["time_series"]
    labels: List[str] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            labels.append(str(item[0]))
        else:
            labels.append(str(item))
    labels = [normalize_marker_label(lab) for lab in labels]
    return mt, labels


def compute_snr(raw_signal, fs):
    """
    SNR = cardiac band power / noise power, in dB.
    
    Bands (explicitly defined for methodological clarity):
        Total band:   0.5 – 8.0 Hz  (physiologically relevant range)
        Cardiac band: 0.5 – 4.0 Hz  (30–240 BPM)
        Noise band:   4.0 – 8.0 Hz  (total minus cardiac)
    
    Pre-processing: bandpass 0.5–8 Hz to remove baseline wander
    and high-frequency sensor noise before PSD estimation.
    """
    TOTAL_LOW, TOTAL_HIGH = 0.5, 8.0
    CARD_LOW, CARD_HIGH = 0.5, 4.0

    # Bandpass filter to isolate physiologically relevant range
    # This removes baseline wander and high-freq noise before PSD
    nyq = fs / 2
    high = min(TOTAL_HIGH, nyq - 0.1)
    sos = sig.butter(4, [TOTAL_LOW, high], btype="bandpass", fs=fs, output="sos")
    x = sig.sosfiltfilt(sos, raw_signal)

    nperseg = min(len(x), int(fs * 10))
    if nperseg < 16:
        return np.nan

    freqs, psd = sig.welch(x, fs=fs, nperseg=nperseg)

    # Explicit band definitions: noise = total - cardiac
    total_mask = (freqs >= TOTAL_LOW) & (freqs <= TOTAL_HIGH)
    cardiac_mask = (freqs >= CARD_LOW) & (freqs <= CARD_HIGH)
    noise_mask = total_mask & ~cardiac_mask

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    p_sig = _trapz(psd[cardiac_mask], freqs[cardiac_mask])
    p_noise = _trapz(psd[noise_mask], freqs[noise_mask])
    if p_noise == 0:
        return float("inf")
    return 10 * np.log10(p_sig / p_noise)


def windowed_snr(signal, fs, window_sec=10):
    """Compute SNR in sliding windows."""
    win_samples = int(window_sec * fs)
    n_windows = len(signal) // win_samples
    snrs = []
    for i in range(n_windows):
        chunk = signal[i * win_samples : (i + 1) * win_samples]
        snrs.append(compute_snr(chunk, fs))
    return np.array(snrs)


def main():
    parser = argparse.ArgumentParser(description="PPG channel selection via SNR for heart rate analysis")
    parser.add_argument("--xdf", required=True, help="Path to .xdf file")
    parser.add_argument("--out", default="analysis_outputs/snr_channel_selection", help="Folder to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print(f"Loading: {args.xdf}")
    streams, _ = pyxdf.load_xdf(args.xdf)

    # --- Markers ---
    marker_stream = find_marker_stream(streams)
    marker_times = None
    marker_labels = None
    if marker_stream:
        marker_times, marker_labels = extract_markers(marker_stream)
        print(f"Marker stream found: {marker_stream['info']['name'][0]}")

    # --- Find EmotiBit PPG streams ---
    idxs = get_emotibit_indices(streams)
    if not idxs:
        print("No EmotiBit_* streams found!")
        return

    print(f"Detected EmotiBits: {idxs}\n")

    for i in idxs:
        ppg_name = f"PPG_EmotiBit_{i}"
        ppg = find_stream_by_name(streams, ppg_name)
        if not ppg:
            print(f"Missing: {ppg_name}")
            continue

        data = np.asarray(ppg["time_series"], dtype=float)
        t = np.asarray(ppg["time_stamps"], dtype=float)
        fs = 1.0 / np.median(np.diff(t))
        n_ch = data.shape[1]
        ch_names = [f"PPG_{ch+1}" for ch in range(n_ch)]

        print(f"{'=' * 55}")
        print(f"Stream: {ppg_name}, Fs={fs:.2f} Hz, Duration={t[-1]-t[0]:.1f}s")
        print(f"Window size: {WINDOW_SEC}s (per Elgendi 2016)")
        print(f"{'=' * 55}")

        # =====================================================
        # 1. FULL RECORDING SNR
        # =====================================================
        print(f"\n{'SNR COMPARISON (median ± IQR across 10s windows)':^55}")
        print(f"  {'Channel':10s} {'Median SNR':>12s} {'IQR':>10s}")
        print("  " + "-" * 35)

        all_snr_windows = {}
        medians = []
        iqrs = []
        for ch in range(n_ch):
            snrs = windowed_snr(data[:, ch], fs, WINDOW_SEC)
            snrs = snrs[np.isfinite(snrs)]
            all_snr_windows[ch_names[ch]] = snrs
            q25, q50, q75 = np.percentile(snrs, [25, 50, 75])
            medians.append(q50)
            iqrs.append(q75 - q25)
            print(f"  {ch_names[ch]:10s} {q50:>10.2f} dB  ±{q75-q25:.2f}")

        best_idx = np.argmax(medians)
        print(f"\n  >>> BEST CHANNEL: {ch_names[best_idx]} ({medians[best_idx]:.2f} dB)")

        # =====================================================
        # 2. PER-SEGMENT SNR
        # =====================================================
        if marker_times is not None and marker_labels is not None:
            print(f"\n{'=' * 55}")
            print("PER-SEGMENT SNR")
            print(f"{'=' * 55}")
            header = f"  {'Segment':30s}"
            for c in ch_names:
                header += f" {c:>8s}"
            header += "  BEST"
            print(header)
            print("  " + "-" * (35 + 10 * n_ch))

            seg_winners = []

            for seg_i in range(len(marker_times)):
                t_start = marker_times[seg_i]
                t_end = marker_times[seg_i + 1] if seg_i + 1 < len(marker_times) else t[-1]
                label = marker_labels[seg_i]

                if (t_end - t_start) < WINDOW_SEC:
                    continue

                mask = (t >= t_start) & (t < t_end)
                if np.sum(mask) < int(fs * WINDOW_SEC):
                    continue

                seg_snrs = []
                for ch in range(n_ch):
                    snrs = windowed_snr(data[mask, ch], fs, WINDOW_SEC)
                    snrs = snrs[np.isfinite(snrs)]
                    med = np.median(snrs) if len(snrs) > 0 else np.nan
                    seg_snrs.append(med)

                bi = np.nanargmax(seg_snrs)
                seg_winners.append(ch_names[bi])

                row = f"  {label:30s}"
                for j, v in enumerate(seg_snrs):
                    marker = " ★" if j == bi else "  "
                    row += f" {v:>6.2f}{marker}"
                row += f"  {ch_names[bi]}"
                print(row)

            from collections import Counter
            counts = Counter(seg_winners)
            overall = counts.most_common(1)[0][0]
            print(f"\n  >>> WINNER ACROSS ALL SEGMENTS: {overall} ({counts[overall]}/{len(seg_winners)} segments)")

        # =====================================================
        # 3. SAVE PLOT
        # =====================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                        gridspec_kw={"width_ratios": [1, 1.5]})
        colors = ["#2196F3", "#FF9800", "#F44336"]

        # LEFT: Bar chart
        bars = ax1.bar(ch_names, medians, yerr=[iq/2 for iq in iqrs],
                       color=colors[:n_ch], edgecolor="black", linewidth=0.5,
                       capsize=8, error_kw={"linewidth": 2})
        bars[best_idx].set_edgecolor("green")
        bars[best_idx].set_linewidth(3)

        for bar, v in zip(bars, medians):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{v:.2f} dB", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_ylabel("SNR (dB)", fontsize=12)
        ax1.set_title("Median SNR across 10s windows", fontsize=12, fontweight="bold")
        ax1.annotate("↑ higher = better (more cardiac power vs noise)",
                    xy=(0.5, 0.02), xycoords="axes fraction", fontsize=9, color="gray", ha="center")

        # RIGHT: Box plot
        box_data = [all_snr_windows[c] for c in ch_names]
        bp = ax2.boxplot(box_data, labels=ch_names, patch_artist=True, widths=0.5,
                         medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors[:n_ch]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        bp["boxes"][best_idx].set_edgecolor("green")
        bp["boxes"][best_idx].set_linewidth(3)

        ax2.set_ylabel("SNR (dB)", fontsize=12)
        ax2.set_title("SNR distribution across all 10s windows", fontsize=12, fontweight="bold")
        ax2.axhline(0, color="gray", linestyle="--", alpha=0.3)

        fig.suptitle(
            f"{ppg_name} — SNR-Based Channel Selection for Heart Rate Analysis\n"
            f"Cardiac: 0.5–4 Hz | Noise: 4–8 Hz | {WINDOW_SEC}s windows | Refs: Elgendi 2016, Charlton et al. 2025",
            fontsize=13, fontweight="bold"
        )
        fig.tight_layout()

        os.makedirs(args.out, exist_ok=True)
        outpath = os.path.join(args.out, f"{ppg_name}_SNR_channel_selection.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {outpath}")

        if args.show:
            plt.show()
        plt.close(fig)

    print(f"\nDone. Check plots in: {args.out}")


if __name__ == "__main__":
    main()

"""
run:
python3 Analysis/01.1_ppg_channel_quality.py \
  --xdf data/raw/session_01.xdf \
  --out analysis_outputs/snr_channel_selection \
  --show
"""