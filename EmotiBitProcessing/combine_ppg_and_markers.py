import pandas as pd
from pathlib import Path

# setup paths
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent / "leftFingerData"
output_dir = script_dir / "output"
output_dir.mkdir(exist_ok=True)

# load PPG CSVs
pi = pd.read_csv(base_dir / "leftFingerData_PI.csv")
pg = pd.read_csv(base_dir / "leftFingerData_PG.csv")
pr = pd.read_csv(base_dir / "leftFingerData_PR.csv")

# load LM file (ensure no extra index column is read)
lm = pd.read_csv(base_dir / "leftFingerData_LM.csv", index_col=False)

# merge based on timestamps
combined_ppg = (
    pi[["LslMarkerSourceTimestamp", "EmotiBitTimestamp", "PI"]]
    .merge(pg[["LslMarkerSourceTimestamp", "EmotiBitTimestamp", "PG"]],
           on=["LslMarkerSourceTimestamp", "EmotiBitTimestamp"], how="outer")
    .merge(pr[["LslMarkerSourceTimestamp", "EmotiBitTimestamp", "PR"]],
           on=["LslMarkerSourceTimestamp", "EmotiBitTimestamp"], how="outer")
    .sort_values("LslMarkerSourceTimestamp")
)

# save combined PPG file
ppg_outfile = output_dir / "leftFingerData_PPG_combined.csv"
combined_ppg.to_csv(ppg_outfile, index=False)
print(f"Combined PPG saved to: {ppg_outfile}")

# detect marker column automatically 
# look for the last column that contains any string (non-numeric) values
marker_col = None
for col in lm.columns[::-1]:
    if lm[col].astype(str).str.contains(r"[A-Za-z]").any():
        marker_col = col
        break

if marker_col is None:
    raise ValueError("No text marker column found in LM file.")

# clean LM file
lm_clean = lm[["LslMarkerSourceTimestamp", "EmotiBitTimestamp", marker_col]].copy()
lm_clean.rename(columns={marker_col: "Marker"}, inplace=True)

# save cleaned LM file
lm_outfile = output_dir / "leftFingerData_LM_cleaned.csv"
lm_clean.to_csv(lm_outfile, index=False)
print(f"Cleaned LM file saved to: {lm_outfile}")