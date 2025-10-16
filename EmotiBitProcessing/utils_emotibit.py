import pandas as pd
import csv

def safe_float(x):
    """Safely convert a string or value to float, or return None if it fails."""
    try:
        return float(x)
    except Exception:
        return None


def parse_lm_file(lm_path):
    """
    Parse EmotiBit *_LM.csv to extract only the LslMarkerSourceTimestamp,
    EmotiBitTimestamp, and Marker columns.
    """
    rows = []
    with open(lm_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row
        for row in reader:
            if not row or len(row) < 9:
                continue
            lsl_src = safe_float(row[0])
            emo_ts  = safe_float(row[3])
            marker  = None

            # Find LD,<marker>
            for i in range(9, len(row) - 1):
                if row[i].strip() == "LD":
                    marker = row[i + 1].strip()
                    break

            # Fallback if LD missing
            if marker is None and len(row) > 9 and any(c.isalpha() for c in row[-1]):
                marker = row[-1].strip()

            if lsl_src is None or emo_ts is None:
                continue

            rows.append({
                "LslMarkerSourceTimestamp": lsl_src,
                "EmotiBitTimestamp": emo_ts,
                "Marker": marker if marker is not None else ""
            })

    return pd.DataFrame(rows)