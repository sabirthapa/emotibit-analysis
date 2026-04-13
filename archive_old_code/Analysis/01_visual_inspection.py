import argparse
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pyxdf


def list_streams(streams: List[dict]) -> None:
    print(f"\nTotal streams found: {len(streams)}\n")
    for i, s in enumerate(streams):
        name = s["info"]["name"][0]
        stype = s["info"]["type"][0]
        srate = s["info"]["nominal_srate"][0]
        nchan = s["info"]["channel_count"][0]
        print(f"{i:02d}: {name:25s} | {stype:10s} | {nchan} ch | {srate} Hz")


def find_stream_by_name(streams: List[dict], name: str) -> Optional[dict]:
    for s in streams:
        if s["info"]["name"][0] == name:
            return s
    return None


def extract_time_series(stream: dict, channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(stream["time_stamps"], dtype=float)
    x = np.asarray(stream["time_series"])
    if x.ndim == 2:
        if channel >= x.shape[1]:
            raise ValueError(f"Channel {channel} out of range for stream with {x.shape[1]} channels.")
        x = x[:, channel]
    else:
        x = x.squeeze()
    return t, x.astype(float)


def find_marker_stream(streams: List[dict]) -> Optional[dict]:
    for s in streams:
        name = str(s["info"]["name"][0]).lower()
        stype = str(s["info"]["type"][0]).lower()
        if "marker" in stype or "marker" in name:
            return s
    return None


def normalize_marker_label(label: str) -> str:
    lab = str(label).strip().lower()
    lab = re.sub(r"\s+", " ", lab)          
    lab = lab.replace(" ", "_")             
    return lab


def sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s


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


def add_marker_lines(ax, marker_times: Optional[np.ndarray], marker_labels: Optional[List[str]]) -> None:
    if marker_times is None or marker_labels is None:
        return
    ymin, ymax = ax.get_ylim()
    for t, lab in zip(marker_times, marker_labels):
        ax.axvline(t, alpha=0.25)
        ax.text(t, ymax, lab, rotation=90, va="top", fontsize=7)


def plot_signal(
    t: np.ndarray,
    x: np.ndarray,
    title: str,
    outpath: Optional[str],
    marker_times: Optional[np.ndarray],
    marker_labels: Optional[List[str]],
    show: bool,
    zoom: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, x)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")

    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1])

    add_marker_lines(ax, marker_times, marker_labels)
    fig.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=150)
        print(f"Saved: {outpath}")

    if show:
        plt.show()

    plt.close(fig)


def plot_ppg_all_channels_stacked(
    ppg_stream: dict,
    title: str,
    outpath: Optional[str],
    marker_times: Optional[np.ndarray],
    marker_labels: Optional[List[str]],
    show: bool,
    zoom: Optional[Tuple[float, float]] = None,
) -> None:
    data = np.asarray(ppg_stream["time_series"])
    t = np.asarray(ppg_stream["time_stamps"], dtype=float)

    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected PPG stream with shape (N,3). Got {data.shape}.")

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    for ch in range(3):
        axes[ch].plot(t, data[:, ch])
        axes[ch].set_ylabel(f"PPG_{ch+1}")

        if zoom is not None:
            axes[ch].set_xlim(zoom[0], zoom[1])

        add_marker_lines(axes[ch], marker_times, marker_labels)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=150)
        print(f"Saved: {outpath}")

    if show:
        plt.show()

    plt.close(fig)


def get_emotibit_indices(streams: List[dict]) -> List[int]:
    idxs = set()
    pattern = re.compile(r"(PPG|EDA|TEMP)_EmotiBit_(\d+)$")
    for s in streams:
        name = s["info"]["name"][0]
        m = pattern.match(name)
        if m:
            idxs.add(int(m.group(2)))
    return sorted(list(idxs))


def marker_zoom_windows(
    marker_times: np.ndarray,
    marker_labels: List[str],
    pad_seconds: float,
) -> List[Tuple[str, Tuple[float, float]]]:
    windows = []
    for t, lab in zip(marker_times, marker_labels):
        z0 = float(t - pad_seconds)
        z1 = float(t + pad_seconds)
        windows.append((lab, (z0, z1)))
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdf", required=True, help="Path to .xdf file")
    parser.add_argument("--out", default="analysis_outputs/visual_inspection", help="Folder to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")

    # PPG options
    parser.add_argument("--ppg_channel", type=int, default=0, help="PPG channel to plot when NOT using --ppg_all_channels (0/1/2)")
    parser.add_argument("--ppg_all_channels", action="store_true", help="Plot all 3 PPG channels stacked (QC)")

    # Fixed-offset zoom (optional)
    parser.add_argument("--fixed_zoom", action="store_true", help="Also save the old fixed-offset zoom plot")
    parser.add_argument("--zoom_seconds", type=float, default=60.0, help="Fixed zoom window length (seconds)")
    parser.add_argument("--zoom_start_offset", type=float, default=300.0, help="Fixed zoom start offset from recording start (seconds)")

    # Marker-based zoom (QC)
    parser.add_argument("--zoom_by_markers", action="store_true", help="Create zoom plots around each marker time")
    parser.add_argument("--marker_pad", type=float, default=15.0, help="Seconds before/after marker for zoom window")

    args = parser.parse_args()

    streams, header = pyxdf.load_xdf(args.xdf)

    print("\n=== STREAM LIST ===")
    list_streams(streams)

    # Markers
    marker_stream = find_marker_stream(streams)
    marker_times = None
    marker_labels = None
    if marker_stream:
        marker_times, marker_labels = extract_markers(marker_stream)
        print(f"\nMarker stream found: {marker_stream['info']['name'][0]} ({marker_stream['info']['type'][0]})")
        print("Markers:", list(zip(marker_times, marker_labels)))
    else:
        print("\nNo marker stream found automatically.")

    idxs = get_emotibit_indices(streams)
    if not idxs:
        print("\nNo EmotiBit_* streams found with expected names. Check stream list output above.")
        return

    print(f"\nDetected EmotiBits: {idxs}")

    # Precompute marker zoom windows
    mz = []
    if args.zoom_by_markers and marker_times is not None and marker_labels is not None:
        mz = marker_zoom_windows(marker_times, marker_labels, args.marker_pad)

    for i in idxs:
        ppg_name = f"PPG_EmotiBit_{i}"
        eda_name = f"EDA_EmotiBit_{i}"
        tmp_name = f"TEMP_EmotiBit_{i}"

        # --- PPG ---
        ppg = find_stream_by_name(streams, ppg_name)
        if ppg:
            t_ppg = np.asarray(ppg["time_stamps"], dtype=float)
            t0 = t_ppg[0]

            # FULL plot
            if args.ppg_all_channels:
                plot_ppg_all_channels_stacked(
                    ppg,
                    title=f"{ppg_name} - ALL CHANNELS (FULL)",
                    outpath=os.path.join(args.out, f"{ppg_name}_allch_full.png"),
                    marker_times=marker_times,
                    marker_labels=marker_labels,
                    show=args.show,
                    zoom=None,
                )
            else:
                t, x = extract_time_series(ppg, channel=args.ppg_channel)
                plot_signal(
                    t, x,
                    title=f"{ppg_name} (ch {args.ppg_channel}) - FULL",
                    outpath=os.path.join(args.out, f"{ppg_name}_ch{args.ppg_channel}_full.png"),
                    marker_times=marker_times,
                    marker_labels=marker_labels,
                    show=args.show,
                )

            # Fixed-offset zoom (optional)
            if args.fixed_zoom:
                z0 = t0 + args.zoom_start_offset
                z1 = z0 + args.zoom_seconds
                zoom = (float(z0), float(z1))

                if args.ppg_all_channels:
                    plot_ppg_all_channels_stacked(
                        ppg,
                        title=f"{ppg_name} - ALL CHANNELS (FIXED ZOOM {args.zoom_seconds:.0f}s @ +{args.zoom_start_offset:.0f}s)",
                        outpath=os.path.join(args.out, f"{ppg_name}_allch_fixedzoom.png"),
                        marker_times=marker_times,
                        marker_labels=marker_labels,
                        show=args.show,
                        zoom=zoom,
                    )
                else:
                    t, x = extract_time_series(ppg, channel=args.ppg_channel)
                    plot_signal(
                        t, x,
                        title=f"{ppg_name} (ch {args.ppg_channel}) - FIXED ZOOM {args.zoom_seconds:.0f}s @ +{args.zoom_start_offset:.0f}s",
                        outpath=os.path.join(args.out, f"{ppg_name}_ch{args.ppg_channel}_fixedzoom.png"),
                        marker_times=marker_times,
                        marker_labels=marker_labels,
                        show=args.show,
                        zoom=zoom,
                    )

            # Marker-based zoom plots (recommended for channel choice)
            for lab, zoom in mz:
                lab_fn = sanitize_filename(lab)

                if args.ppg_all_channels:
                    plot_ppg_all_channels_stacked(
                        ppg,
                        title=f"{ppg_name} - ALL CHANNELS (ZOOM around {lab}, ±{args.marker_pad:.0f}s)",
                        outpath=os.path.join(args.out, f"{ppg_name}_allch_zoom_{lab_fn}.png"),
                        marker_times=marker_times,
                        marker_labels=marker_labels,
                        show=args.show,
                        zoom=zoom,
                    )
                else:
                    t, x = extract_time_series(ppg, channel=args.ppg_channel)
                    plot_signal(
                        t, x,
                        title=f"{ppg_name} (ch {args.ppg_channel}) - ZOOM around {lab}, ±{args.marker_pad:.0f}s",
                        outpath=os.path.join(args.out, f"{ppg_name}_ch{args.ppg_channel}_zoom_{lab_fn}.png"),
                        marker_times=marker_times,
                        marker_labels=marker_labels,
                        show=args.show,
                        zoom=zoom,
                    )
        else:
            print(f"Missing: {ppg_name}")

        # --- EDA ---
        eda = find_stream_by_name(streams, eda_name)
        if eda:
            t, x = extract_time_series(eda, channel=0)
            plot_signal(
                t, x,
                title=f"{eda_name} - FULL",
                outpath=os.path.join(args.out, f"{eda_name}_full.png"),
                marker_times=marker_times,
                marker_labels=marker_labels,
                show=args.show,
            )
        else:
            print(f"Missing: {eda_name}")

        # --- TEMP ---
        tmp = find_stream_by_name(streams, tmp_name)
        if tmp:
            t, x = extract_time_series(tmp, channel=0)
            plot_signal(
                t, x,
                title=f"{tmp_name} - FULL",
                outpath=os.path.join(args.out, f"{tmp_name}_full.png"),
                marker_times=marker_times,
                marker_labels=marker_labels,
                show=args.show,
            )
        else:
            print(f"Missing: {tmp_name}")

    print("\nDone. Check plots in:", args.out)
    if not args.show:
        print("Tip: rerun with --show to pop up plots interactively.")


if __name__ == "__main__":
    main()

"""
run:
python3 analysis/01_visual_inspection.py \
  --xdf data/raw/session_01.xdf \
  --ppg_all_channels
"""