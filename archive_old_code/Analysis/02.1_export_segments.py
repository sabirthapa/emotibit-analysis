import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyxdf


def find_stream_by_name(streams: List[dict], name: str) -> Optional[dict]:
    for s in streams:
        if s["info"]["name"][0] == name:
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


def extract_stream_ts(stream: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t, x). x is ndarray: (N,) for 1ch OR (N,C) for multich."""
    t = np.asarray(stream["time_stamps"], dtype=float)
    x = np.asarray(stream["time_series"], dtype=float)
    # pyxdf typically gives (N, C) for multichannel; keep as-is
    return t, x


def slice_by_time(t: np.ndarray, x: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (t >= t0) & (t <= t1)
    return t[mask], x[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", required=True, help="Path to .xdf file")
    ap.add_argument(
        "--segments_csv",
        default="analysis_outputs/segments/session_segments.csv",
        help="CSV from segmentation step",
    )
    ap.add_argument("--out", default="analysis_outputs/segmented", help="Output root folder")
    args = ap.parse_args()

    # Fixed channel picks you confirmed
    # Keys are EmotiBit index: chosen PPG channel (0-based)
    ppg_channel_pick: Dict[int, int] = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 2,
    }

    # Load segments
    seg_df = pd.read_csv(args.segments_csv)
    # Keep everything except post_meditation (if it exists)
    seg_df = seg_df[seg_df["segment"] != "post_meditation"].reset_index(drop=True)
    print("Keeping segments:", seg_df["segment"].tolist())
    required_cols = {"segment", "t_start", "t_end"}
    if not required_cols.issubset(set(seg_df.columns)):
        raise ValueError(f"segments_csv missing required cols: {required_cols}")

    # Load XDF
    print(f"Loading XDF: {args.xdf}")
    streams, _ = pyxdf.load_xdf(args.xdf)

    idxs = get_emotibit_indices(streams)
    if not idxs:
        raise RuntimeError("No EmotiBit streams found (expected names like PPG_EmotiBit_1).")

    os.makedirs(args.out, exist_ok=True)

    print(f"Found EmotiBits: {idxs}")
    print(f"Segments loaded: {len(seg_df)}")

    for eb in idxs:
        if eb not in ppg_channel_pick:
            print(f"WARNING: No PPG channel pick provided for EmotiBit_{eb}; skipping PPG export.")
        eb_dir = os.path.join(args.out, f"EmotiBit_{eb}")
        os.makedirs(eb_dir, exist_ok=True)

        # Stream names
        ppg_name = f"PPG_EmotiBit_{eb}"
        eda_name = f"EDA_EmotiBit_{eb}"
        tmp_name = f"TEMP_EmotiBit_{eb}"

        ppg_stream = find_stream_by_name(streams, ppg_name)
        eda_stream = find_stream_by_name(streams, eda_name)
        tmp_stream = find_stream_by_name(streams, tmp_name)

        # Load arrays once per stream (efficient)
        ppg_t = ppg_x = None
        if ppg_stream is not None:
            ppg_t, ppg_x = extract_stream_ts(ppg_stream)
        eda_t = eda_x = None
        if eda_stream is not None:
            eda_t, eda_x = extract_stream_ts(eda_stream)
        tmp_t = tmp_x = None
        if tmp_stream is not None:
            tmp_t, tmp_x = extract_stream_ts(tmp_stream)

        for _, row in seg_df.iterrows():
            seg_name = str(row["segment"])
            t0 = float(row["t_start"])
            t1 = float(row["t_end"])
            seg_dir = os.path.join(eb_dir, seg_name)
            os.makedirs(seg_dir, exist_ok=True)

            # PPG (selected channel only)
            if ppg_t is not None and ppg_x is not None and eb in ppg_channel_pick:
                ch = ppg_channel_pick[eb]
                # ppg_x should be (N, C)
                if ppg_x.ndim != 2 or ch >= ppg_x.shape[1]:
                    print(f"WARNING: {ppg_name} unexpected shape {ppg_x.shape}; skipping PPG export for this segment.")
                else:
                    t_seg, x_seg = slice_by_time(ppg_t, ppg_x[:, ch], t0, t1)
                    out_csv = os.path.join(seg_dir, "PPG.csv")
                    pd.DataFrame({"time": t_seg, "ppg": x_seg}).to_csv(out_csv, index=False)

            # EDA
            if eda_t is not None and eda_x is not None:
                # eda_x likely (N,) or (N,1)
                if eda_x.ndim == 2:
                    eda_sig = eda_x[:, 0]
                else:
                    eda_sig = eda_x
                t_seg, x_seg = slice_by_time(eda_t, eda_sig, t0, t1)
                out_csv = os.path.join(seg_dir, "EDA.csv")
                pd.DataFrame({"time": t_seg, "eda": x_seg}).to_csv(out_csv, index=False)

            # TEMP
            if tmp_t is not None and tmp_x is not None:
                if tmp_x.ndim == 2:
                    tmp_sig = tmp_x[:, 0]
                else:
                    tmp_sig = tmp_x
                t_seg, x_seg = slice_by_time(tmp_t, tmp_sig, t0, t1)
                out_csv = os.path.join(seg_dir, "TEMP.csv")
                pd.DataFrame({"time": t_seg, "temp": x_seg}).to_csv(out_csv, index=False)

        print(f"Exported segments for EmotiBit_{eb}")

    print(f"\nDone. Output root: {args.out}")

if __name__ == "__main__":
    main()

"""
python3 analysis/02.1_export_segments.py \
  --xdf data/raw/session_01.xdf \
  --segments_csv analysis_outputs/segments_v2/session_segments_v2.csv \
  --out analysis_outputs/segmented_v2
"""