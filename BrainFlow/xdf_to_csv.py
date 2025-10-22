import pyxdf
import pandas as pd
import os

# path to your XDF file
xdf_path = "/Users/sabirthapa/Documents/CurrentStudy/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"

print(f"Loading XDF file: {xdf_path}")
streams, fileheader = pyxdf.load_xdf(xdf_path)
print(f"Loaded {len(streams)} streams.")

# create an output folder for CSVs
output_dir = os.path.join(os.path.dirname(xdf_path), "csv_exports")
os.makedirs(output_dir, exist_ok=True)

# convert each stream to CSV
for i, stream in enumerate(streams):
    name = stream["info"]["name"][0]
    data = stream["time_series"]
    timestamps = stream["time_stamps"]

    # Build dataframe
    cols = []
    if "PPG" in name:
        cols = ["PPG_1", "PPG_2", "PPG_3"]
    elif "EDA" in name:
        cols = ["EDA"]
    elif "TEMP" in name:
        cols = ["Temperature"]
    else:
        cols = [f"Ch_{j}" for j in range(data.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    df["Timestamp"] = timestamps

    # save as CSV
    out_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

print("\nAll streams exported successfully to:", output_dir)