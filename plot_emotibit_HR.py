import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = {
    "Finger": "Finger-099/first_HR.csv",
    "Wrist": "Wrist-258/first_HR.csv",
    "Arm":   "Arms-228/first_HR.csv"
}

skip_sec = 2
signals = {}
time_axes = {}
hr_stats = {}

print("\n=== Heart Rate (HR, bpm) ===")

for name, path in files.items():
    df = pd.read_csv(path)

    if "LocalTimestamp" not in df.columns or "HR" not in df.columns:
        print(f"⚠️ Missing HR data in {path}")
        continue

    ts = df["LocalTimestamp"].astype(float).values
    sig = df["HR"].astype(float).values
    ts = ts - ts[0]

    mask = ts >= skip_sec
    ts, sig = ts[mask], sig[mask]

    if sig.size == 0:
        print(f"⚠️ {name}: No HR samples after trimming")
        continue

    mean_hr = np.mean(sig)
    median_hr = np.median(sig)
    hr_stats[name] = (mean_hr, median_hr)

    print(f"{name}: mean={mean_hr:.1f} bpm, median={median_hr:.1f} bpm")

    signals[name] = sig
    time_axes[name] = ts

# --- Plot scatter of HR over time ---
plt.figure(figsize=(12,6))
for name in signals:
    plt.scatter(time_axes[name], signals[name], label=f"{name}")

plt.title("Heart Rate (HR, bpm) – Finger vs Wrist vs Arm (Discrete Samples)")
plt.xlabel("Time (s)")
plt.ylabel("HR (bpm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("HR_scatter.png", dpi=300)
plt.show()

# --- Plot bar summary (mean & median) ---
labels = list(hr_stats.keys())
means = [hr_stats[n][0] for n in labels]
medians = [hr_stats[n][1] for n in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, means, width, label="Mean HR")
plt.bar(x + width/2, medians, width, label="Median HR")

plt.xticks(x, labels)
plt.ylabel("HR (bpm)")
plt.title("Summary of Heart Rate (Mean vs Median)")
plt.legend()
plt.tight_layout()
plt.savefig("HR_summary.png", dpi=300)
plt.show()