# plot_emotibit_SCR.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Files (per device) ----
FILES = {
    "Finger": {
        "SA": "Finger-099/first_SA.csv",   # SCR amplitude (µS)
        "SF": "Finger-099/first_SF.csv",   # SCR frequency (count/min)
        "SR": "Finger-099/first_SR.csv",   # rise time (s)
    },
    "Wrist": {
        "SA": "Wrist-258/first_SA.csv",
        "SF": "Wrist-258/first_SF.csv",
        "SR": "Wrist-258/first_SR.csv",
    },
    "Arm": {
        "SA": "Arms-228/first_SA.csv",
        "SF": "Arms-228/first_SF.csv",
        "SR": "Arms-228/first_SR.csv",
    },
}

FS_HINT = { "SA": 15, "SF": 3, "SR": 3 }  # nominal rates (not strictly needed)
SKIP_SEC = 2     # ignore first X seconds
LABELS = {"SA":"SCR Amplitude (µS)", "SF":"SCR Frequency (count/min)", "SR":"SCR Rise Time (s)"}

def load_channel(path, value_col_name):
    df = pd.read_csv(path)
    if "LocalTimestamp" not in df.columns or value_col_name not in df.columns:
        return None, None
    ts = df["LocalTimestamp"].astype(float).values
    val = pd.to_numeric(df[value_col_name], errors="coerce").values
    # drop nans safely
    mask = np.isfinite(ts) & np.isfinite(val)
    ts, val = ts[mask], val[mask]
    if ts.size == 0:
        return None, None
    # make relative time and trim start
    ts = ts - ts[0]
    keep = ts >= SKIP_SEC
    return ts[keep], val[keep]

def zscore(x):
    m, s = np.mean(x), np.std(x)
    if s == 0 or not np.isfinite(s):
        return x - m
    return (x - m) / s

# ---- Load everything ----
data = {ch: {} for ch in ["SA","SF","SR"]}   # data["SA"]["Finger"] = (ts, values)
stats = {ch: {} for ch in ["SA","SF","SR"]}  # simple stats to print + legend

for ch in ["SA","SF","SR"]:
    for site, paths in FILES.items():
        ts, val = load_channel(paths[ch], ch)
        if ts is None or val is None or val.size == 0:
            print(f"⚠️ Missing/empty {ch} for {site}: {paths[ch]}")
            continue
        data[ch][site] = (ts, val)
        if ch == "SA":
            stats[ch][site] = {"mean": np.nanmean(val), "median": np.nanmedian(val)}
        elif ch == "SF":
            stats[ch][site] = {"mean": np.nanmean(val)}  # counts/min
        else:  # SR
            stats[ch][site] = {"median": np.nanmedian(val)}

# ---- Align and plot helper ----
def plot_channel(ch, title, ylabel, outfile):
    if not data[ch]:
        print(f"❌ No data for {ch}, skipping plot.")
        return
    # overlapping time window across all present sites
    common_start = max(v[0][0] for v in data[ch].values())
    common_end   = min(v[0][-1] for v in data[ch].values())

    plt.figure(figsize=(12,6))
    for site, (ts, val) in data[ch].items():
        mask = (ts >= common_start) & (ts <= common_end)
        ts_cut, val_cut = ts[mask], val[mask]
        if ts_cut.size == 0:
            continue
        # normalize for visual comparability (unitless)
        val_norm = zscore(val_cut)

        # Legend text with compact stats
        if ch == "SA":
            leg = f"{site} (μ={stats[ch][site]['mean']:.2f} μS, med={stats[ch][site]['median']:.2f} μS)"
        elif ch == "SF":
            leg = f"{site} (mean={stats[ch][site]['mean']:.2f} cpm)"
        else:  # SR
            leg = f"{site} (median={stats[ch][site]['median']:.2f} s)"

        plt.plot(ts_cut, val_norm, label=leg)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized " + ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()
    print(f"✅ Saved {outfile}")

# ---- Make the 3 comparison plots ----
plot_channel("SA",
             "SCR Amplitude – Finger vs Wrist vs Arm (Aligned, Normalized)",
             LABELS["SA"],
             "SCR_amplitude_comparison.png")

plot_channel("SF",
             "SCR Frequency – Finger vs Wrist vs Arm (Aligned, Normalized)",
             LABELS["SF"],
             "SCR_frequency_comparison.png")

plot_channel("SR",
             "SCR Rise Time – Finger vs Wrist vs Arm (Aligned, Normalized)",
             LABELS["SR"],
             "SCR_risetime_comparison.png")

# ---- Console summary table ----
print("\n=== SCR Summary (per site) ===")
for ch in ["SA","SF","SR"]:
    if not stats[ch]:
        continue
    print(f"\n{ch}:")
    for site in FILES.keys():
        if site not in stats[ch]:
            print(f"  {site}: n/a")
            continue
        s = stats[ch][site]
        if ch == "SA":
            print(f"  {site}: mean={s['mean']:.3f} μS, median={s['median']:.3f} μS")
        elif ch == "SF":
            print(f"  {site}: mean={s['mean']:.3f} count/min")
        else:
            print(f"  {site}: median={s['median']:.3f} s")