# analysisV2/02_segmentation_splitAB.py  (PASTABLE FULL FILE)
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

# markers we expect (we also add synthetic "end" later)
BOUNDARY_KEYS = ["baseline", "start", "stop"]


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


def get_first_boundary_times(marker_times, marker_labels):
    boundary_time = {}
    for t, lab in zip(marker_times, marker_labels):
        if lab in (BOUNDARY_KEYS + ["meditation", "quiet", "eyes_open"]) and lab not in boundary_time:
            boundary_time[lab] = float(t)

    missing = [k for k in BOUNDARY_KEYS if k not in boundary_time]
    if missing:
        raise ValueError(f"Missing required markers: {missing}. Found: {sorted(set(marker_labels))}")

    return boundary_time


def build_segments_split_ab(boundary_time, end_time, buffer_s=5.0, min_len_s=10.0, include_post=False):
    """
    baseline_warmup: baseline -> start
    meditation_2A:   first half of (start -> stop) after trimming buffer
    meditation_2B:   second half of (start -> stop) after trimming buffer
    optional post_meditation: stop -> end
    """
    boundary_time = dict(boundary_time)
    boundary_time["end"] = float(end_time)

    # ---- baseline warmup ----
    base0 = boundary_time["baseline"] + buffer_s
    base1 = boundary_time["start"] - buffer_s

    # ---- meditation trimmed interval ----
    med0 = boundary_time["start"] + buffer_s
    med1 = boundary_time["stop"] - buffer_s
    med_dur = med1 - med0
    if med_dur < (2 * min_len_s):
        raise ValueError(
            f"Meditation duration too short after trimming: {med_dur:.2f}s. "
            f"Try smaller --buffer_s or lower --min_len_s."
        )

    # split midpoint
    mid = med0 + 0.5 * med_dur
    a0, a1 = med0, mid
    b0, b1 = mid, med1

    seg_defs = [
        ("baseline_warmup", base0, base1, "baseline", "start"),
        ("meditation_2A", a0, a1, "start", "midpoint"),
        ("meditation_2B", b0, b1, "midpoint", "stop"),
    ]

    if include_post:
        post0 = boundary_time["stop"] + buffer_s
        post1 = boundary_time["end"] - buffer_s
        seg_defs.append(("post_meditation", post0, post1, "stop", "end"))

    rows = []
    for seg_name, t0, t1, a, b in seg_defs:
        dur = t1 - t0
        if dur < min_len_s:
            # skip too-short segments
            continue
        rows.append(
            {
                "segment": seg_name,
                "t_start": float(t0),
                "t_end": float(t1),
                "duration_s": float(dur),
                "start_marker": a,
                "end_marker": b,
                "buffer_s": float(buffer_s),
            }
        )

    df = pd.DataFrame(rows)
    return df, boundary_time, float(mid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdf", required=True)
    ap.add_argument("--out", default="analysis_outputs/segments_v2")
    ap.add_argument("--buffer_s", type=float, default=5.0)
    ap.add_argument("--min_len_s", type=float, default=10.0)
    ap.add_argument("--include_post", action="store_true", help="Also create post_meditation = stop->end")
    args = ap.parse_args()

    streams, _ = pyxdf.load_xdf(args.xdf)
    ms = find_marker_stream(streams)
    if ms is None:
        raise RuntimeError("No marker stream found in XDF.")

    # recording end time = max timestamp across all streams
    end_time = max(float(np.max(s["time_stamps"])) for s in streams if len(s.get("time_stamps", [])) > 0)

    mt, ml = extract_markers(ms)
    boundary_time = get_first_boundary_times(mt, ml)

    df, boundary_time, mid = build_segments_split_ab(
        boundary_time=boundary_time,
        end_time=end_time,
        buffer_s=args.buffer_s,
        min_len_s=args.min_len_s,
        include_post=args.include_post,
    )

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "session_segments_v2.csv")
    df.to_csv(out_csv, index=False)

    print("Marker times (used):")
    for k in ["baseline", "start", "stop", "end"]:
        print(f"  {k:12s} -> {boundary_time[k]:.3f}")
    print(f"  {'midpoint':12s} -> {mid:.3f}   (split of trimmed meditation interval)")

    print(f"\nSaved segments CSV: {out_csv}")
    print(df)


if __name__ == "__main__":
    main()

# run:
# python3 analysisV2/02_segmentation_splitAB.py --xdf data/raw/session_01.xdf --buffer_s 5
# optional:
# python3 analysisV2/02_segmentation_splitAB.py --xdf data/raw/session_01.xdf --buffer_s 5 --include_post

""" 
export segments using new csv
run:
python3 analysis/02.1_export_segments.py \
  --xdf data/raw/session_01.xdf \
  --segments_csv analysis_outputs/segments_v2/session_segments_v2.csv \
  --out analysis_outputs/segmented_v2

  
HR extraction
run:
python3 analysis/03_ppg_hr_extraction.py \
  --seg_root analysis_outputs/segmented_v2 \
  --out_root analysis_outputs/hr_raw_v2 \
  --max_gap_s 2.5

clean HR extraction
run:
python3 analysis/03.1_clean_hr_series.py \
  --hr_root analysis_outputs/hr_raw_v2 \
  --out_root analysis_outputs/hr_clean_v2 \
  --max_jump 15 \
  --max_gap_s 5 \
  --smooth_s 5

Synchrony (now using new segments)
run:
python3 analysis/04_hr_synchrony.py \
  --hr_root analysis_outputs/hr_clean_v2 \
  --out_root analysis_outputs/synchrony_v2_lag5 \
  --segments baseline_warmup,meditation_2A,meditation_2B \
  --window_s 30 --step_s 1 \
  --max_lag_s 5
"""