import pandas as pd
from pathlib import Path

# setup paths
script_dir = Path(__file__).resolve().parent
data_root = script_dir.parent  # parent folder that contains subject1, subject2, etc.
output_dir = script_dir / "output"
output_dir.mkdir(exist_ok=True)

# find all subject folders dynamically (those starting with "subject")
subject_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("subject")]

for subject_dir in subject_dirs:
    subject_name = subject_dir.name
    print(f"\nProcessing {subject_name}...")

    try:
        # load PPG CSVs
        pi = pd.read_csv(subject_dir / f"{subject_name}_PI.csv")
        pg = pd.read_csv(subject_dir / f"{subject_name}_PG.csv")
        pr = pd.read_csv(subject_dir / f"{subject_name}_PR.csv")

        # load LM CSV
        lm = pd.read_csv(subject_dir / f"{subject_name}_LM.csv", index_col=False)

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
        ppg_outfile = output_dir / f"{subject_name}_PPG_combined.csv"
        combined_ppg.to_csv(ppg_outfile, index=False)
        print(f"Combined PPG saved to: {ppg_outfile}")

        # detect marker column automatically
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
        lm_outfile = output_dir / f"{subject_name}_LM_cleaned.csv"
        lm_clean.to_csv(lm_outfile, index=False)
        print(f"Cleaned LM file saved to: {lm_outfile}")

    except FileNotFoundError as e:
        print(f"Skipping {subject_name}: Missing file -> {e}")
    except Exception as e:
        print(f"Error processing {subject_name}: {e}")

print("\nAll subjects processed.")