import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

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
    """Returns (t, x). If multi-channel, selects `channel`."""
    t = np.asarray(stream["time_stamps"], dtype=float)
    x = np.asarray(stream["time_series"])
    if x.ndim == 2:
        # x shape: (samples, channels)
        if channel >= x.shape[1]:
            raise ValueError(f"Channel {channel} out of range for stream with {x.shape[1]} channels.")
        x = x[:, channel]
    else:
        x = x.squeeze()
    return t, x.astype(float)


def find_marker_stream(streams: List[dict]) -> Optional[dict]:
    """
    Try to locate a marker stream.
    Many setups store markers as type 'Markers' or similar.
    We'll search for type containing 'marker' OR name containing 'marker'.
    """
    for s in streams:
        name = str(s["info"]["name"][0]).lower()
        stype = str(s["info"]["type"][0]).lower()
        if "marker" in stype or "marker" in name:
            return s
    return None


def extract_markers(marker_stream: dict) -> Tuple[np.ndarray, List[str]]:
    mt = np.asarray(marker_stream["time_stamps"], dtype=float)
    # time_series is typically list of lists like [[label],[label],...]
    raw = marker_stream["time_series"]
    labels: List[str] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            labels.append(str(item[0]))
        else:
            labels.append(str(item))
    return mt, labels


def add_marker_lines(ax, marker_times: Optional[np.ndarray], marker_labels: Optional[List[str]]) -> None:
    if marker_times is None or marker_labels is None:
        return
    ymin, ymax = ax.get_ylim()
    for t, lab in zip(marker_times, marker_labels):
        ax.axvline(t, alpha=0.25)
        # Put small text near top
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


def get_emotibit_indices(streams: List[dict]) -> List[int]:
    """
    From stream names like PPG_EmotiBit_1, EDA_EmotiBit_2, etc.,
    collect all indices.
    """
    idxs = set()
    pattern = re.compile(r"(PPG|EDA|TEMP)_EmotiBit_(\d+)$")
    for s in streams:
        name = s["info"]["name"][0]
        m = pattern.match(name)
        if m:
            idxs.add(int(m.group(2)))
    return sorted(list(idxs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdf", required=True, help="Path to .xdf file")
    parser.add_argument("--out", default="analysis_outputs/visual_inspection", help="Folder to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--ppg_channel", type=int, default=0, help="Which PPG channel to plot (0/1/2)")
    parser.add_argument("--zoom_seconds", type=float, default=60.0, help="Zoom window length (seconds)")
    parser.add_argument("--zoom_start_offset", type=float, default=300.0, help="Zoom start offset from recording start (seconds)")
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
        print("First markers:", list(zip(marker_times[:10], marker_labels[:10])))
    else:
        print("\nNo marker stream found automatically. (That’s OK; we can add it once you tell me the stream name.)")

    # Which EmotiBits exist?
    idxs = get_emotibit_indices(streams)
    if not idxs:
        print("\nNo EmotiBit_* streams found with expected names. Check stream list output above.")
        return

    print(f"\nDetected EmotiBits: {idxs}")

    # For each EmotiBit: plot full and zoomed for PPG, EDA, TEMP
    for i in idxs:
        ppg_name = f"PPG_EmotiBit_{i}"
        eda_name = f"EDA_EmotiBit_{i}"
        tmp_name = f"TEMP_EmotiBit_{i}"

        # --- PPG ---
        ppg = find_stream_by_name(streams, ppg_name)
        if ppg:
            t, x = extract_time_series(ppg, channel=args.ppg_channel)
            title = f"{ppg_name} (ch {args.ppg_channel}) - FULL"
            plot_signal(
                t, x, title,
                outpath=os.path.join(args.out, f"{ppg_name}_ch{args.ppg_channel}_full.png"),
                marker_times=marker_times, marker_labels=marker_labels,
                show=args.show
            )

            # zoom
            t0 = t[0]
            z0 = t0 + args.zoom_start_offset
            z1 = z0 + args.zoom_seconds
            title = f"{ppg_name} (ch {args.ppg_channel}) - ZOOM {args.zoom_seconds:.0f}s @ +{args.zoom_start_offset:.0f}s"
            plot_signal(
                t, x, title,
                outpath=os.path.join(args.out, f"{ppg_name}_ch{args.ppg_channel}_zoom.png"),
                marker_times=marker_times, marker_labels=marker_labels,
                show=args.show,
                zoom=(z0, z1)
            )
        else:
            print(f"Missing: {ppg_name}")

        # --- EDA ---
        eda = find_stream_by_name(streams, eda_name)
        if eda:
            t, x = extract_time_series(eda, channel=0)
            title = f"{eda_name} - FULL"
            plot_signal(
                t, x, title,
                outpath=os.path.join(args.out, f"{eda_name}_full.png"),
                marker_times=marker_times, marker_labels=marker_labels,
                show=args.show
            )
        else:
            print(f"Missing: {eda_name}")

        # --- TEMP ---
        tmp = find_stream_by_name(streams, tmp_name)
        if tmp:
            t, x = extract_time_series(tmp, channel=0)
            title = f"{tmp_name} - FULL"
            plot_signal(
                t, x, title,
                outpath=os.path.join(args.out, f"{tmp_name}_full.png"),
                marker_times=marker_times, marker_labels=marker_labels,
                show=args.show
            )
        else:
            print(f"Missing: {tmp_name}")

    print("\nDone. Check plots in:", args.out)
    if not args.show:
        print("Tip: rerun with --show to pop up plots interactively.")


if __name__ == "__main__":
    main()