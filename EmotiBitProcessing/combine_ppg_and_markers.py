import pandas as pd
from pathlib import Path
import csv
from utils_emotibit import parse_lm_file, safe_float

# Define paths
script_dir = Path(__file__).resolve().parent
data_root  = script_dir.parent            # contains subject1, subject2, ...
output_dir = script_dir / "output"        # output inside EmotiBitProcessing/
output_dir.mkdir(exist_ok=True)

# subject folders like subject1, subject2, ...
subject_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("subject")]

for subject_dir in subject_dirs:
    subject_name = subject_dir.name
    print(f"\nProcessing {subject_name}...")

    try:
        # load PPG CSVs 
        pi = pd.read_csv(subject_dir / f"{subject_name}_PI.csv")
        pg = pd.read_csv(subject_dir / f"{subject_name}_PG.csv")
        pr = pd.read_csv(subject_dir / f"{subject_name}_PR.csv")

        # merge PPG files on timestamps
        combined_ppg = (
            pi[["LslMarkerSourceTimestamp", "EmotiBitTimestamp", "PI"]]
            .merge(
                pg[["LslMarkerSourceTimestamp", "EmotiBitTimestamp", "PG"]],
                on=["LslMarkerSourceTimestamp", "EmotiBitTimestamp"], how="outer"
            )
            .merge(
                pr[["LslMarkerSourceTimestamp", "EmotiBitTimestamp", "PR"]],
                on=["LslMarkerSourceTimestamp", "EmotiBitTimestamp"], how="outer"
            )
            .sort_values("LslMarkerSourceTimestamp")
        )

        # format numeric columns
        # Keep EmotiBitTimestamp with 3 decimal places
        if "EmotiBitTimestamp" in combined_ppg.columns:
            combined_ppg["EmotiBitTimestamp"] = combined_ppg["EmotiBitTimestamp"].apply(
                lambda x: f"{float(x):.3f}" if pd.notna(x) else ""
            )

        # Format PI, PG, PR as integers (remove .0)
        for col in ["PI", "PG", "PR"]:
            if col in combined_ppg.columns:
                combined_ppg[col] = combined_ppg[col].apply(
                    lambda x: str(int(x)) if pd.notna(x) and float(x).is_integer() else ("" if pd.isna(x) else str(x))
                )

        # save combined PPG file
        ppg_outfile = output_dir / f"{subject_name}_PPG_combined.csv"
        combined_ppg.to_csv(ppg_outfile, index=False)
        print(f"Combined PPG saved to: {ppg_outfile}")

        # parse LM robustly 
        lm_path = subject_dir / f"{subject_name}_LM.csv"
        lm_clean = parse_lm_file(lm_path)

        lm_outfile = output_dir / f"{subject_name}_LM_cleaned.csv"
        lm_clean.to_csv(lm_outfile, index=False)
        print(f"Cleaned LM file saved to: {lm_outfile}")

        if not lm_clean.empty:
            head = lm_clean.head(3).to_string(index=False)
            print("LM preview:\n" + head)

    except FileNotFoundError as e:
        print(f"Skipping {subject_name}: Missing file -> {e}")
    except Exception as e:
        print(f"Error processing {subject_name}: {e}")

print("\nAll subjects processed.")