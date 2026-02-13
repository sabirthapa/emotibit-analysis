# analysis/02_segmentation.py  (PASTABLE FULL FILE)
import argparse
import os
import numpy as np
import pandas as pd
import pyxdf

CANON = {
    "baseline": "baseline",
    "start": "start",
    "longer version meditation": "meditation",
    "longer version meditation ": "meditation",
    "longer version meditation  ": "meditation",
    "quiet session": "quiet",
    "eyes open": "eyes_open",
    "eyes open ": "eyes_open",
    "eyes open  ": "eyes_open",
    "stop": "stop",
}

# ordered boundaries we expect (plus synthetic "end" we create from stream end time)
BOUNDARY_ORDER = ["baseline", "start", "meditation", "quiet", "eyes_open", "stop", "end"]


def find_marker_stream(streams):
    for s in streams:
        name = str(s["info"]["name"][0]).lower()
        stype = str(s["info"]["type"][0]).lower()
        if "marker" in stype or "marker" in name:
            return s
    return None


def extract_markers(marker_stream):
    mt = np.asarray(marker_stream["time_stamps"], dtype=float)
    raw = marker_stream["time_series"]
    labels = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) > 0:
            labels.append(str(item[0]))
        else:
            labels.append(str(item))
    labels = [str(x).strip().lower() for x in labels]
    labels = [CANON.get(lab, lab) for lab in labels]
    return mt, labels


def build_segments(marker_times, marker_labels, end_time, buffer_s=5.0, min_len_s=10.0):
    # Keep only first occurrence of each boundary label in desired order
    boundary_time = {}
    for t, lab in zip(marker_times, marker_labels):
        if lab in BOUNDARY_ORDER and lab not in boundary_time:
            boundary_time[lab] = float(t)

    missing = [b for b in ["baseline", "start", "stop"] if b not in boundary_time]
    if missing:
        raise ValueError(f"Missing required markers: {missing}. Found: {sorted(set(marker_labels))}")

    # Add synthetic end boundary from recording end time
    boundary_time["end"] = float(end_time)

    # You want:
    # - baseline warmup = baseline -> start
    # - meditation (concatenated) = start -> stop  (includes longer meditation + quiet + eyes open)
    # - post meditation = stop -> end
    seg_defs = [
        ("baseline_warmup", "baseline", "start"),
        ("meditation_all", "start", "stop"),
        ("post_meditation", "stop", "end"),
    ]

    rows = []
    for seg_name, a, b in seg_defs:
        t0 = boundary_time[a] + buffer_s
        t1 = boundary_time[b] - buffer_s
        dur = t1 - t0
        if dur < min_len_s:
            # skip too-short segments after trimming
            continue
        rows.append(
            {
                "segment": seg_name,
                "t_start": t0,
                "t_end": t1,
                "duration_s": dur,
                "start_marker": a,
                "end_marker": b,
                "buffer_s": buffer_s,
            }
        )

    return pd.DataFrame(rows), boundary_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", required=True)
    ap.add_argument("--out", default="analysis_outputs/segments")
    ap.add_argument("--buffer_s", type=float, default=5.0)
    args = ap.parse_args()

    streams, _ = pyxdf.load_xdf(args.xdf)
    ms = find_marker_stream(streams)
    if ms is None:
        raise RuntimeError("No marker stream found in XDF.")

    # Recording end time (max timestamp across all streams)
    end_time = max(
        float(np.max(s["time_stamps"])) for s in streams if len(s.get("time_stamps", [])) > 0
    )

    mt, ml = extract_markers(ms)
    df, boundary_time = build_segments(mt, ml, end_time=end_time, buffer_s=args.buffer_s)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "session_segments.csv")
    df.to_csv(out_csv, index=False)

    print("Marker times (used):")
    for k in ["baseline", "start", "meditation", "quiet", "eyes_open", "stop", "end"]:
        if k in boundary_time:
            print(f"  {k:12s} -> {boundary_time[k]:.3f}")

    print(f"\nSaved segments CSV: {out_csv}")
    print(df)


if __name__ == "__main__":
    main()

# run:
# python3 analysis/02_segmentation.py --xdf data/raw/session_01.xdf --buffer_s 5