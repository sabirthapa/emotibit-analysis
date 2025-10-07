import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, detrend

# --- Bandpass filter for EDA (slow varying: 0.05–2 Hz) ---
def bandpass_filter(signal, fs=15, low=0.05, high=2, order=3):
    nyquist = 0.5 * fs
    b, a = butter(order, [low/nyquist, high/nyquist], btype='band')
    return filtfilt(b, a, signal)

# --- SNR calculation ---
def compute_snr(sig):
    power_signal = np.mean(sig**2)
    noise = sig - np.convolve(sig, np.ones(10)/10, mode='same')  # smooth baseline
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / power_noise) if power_noise > 0 else -np.inf

# --- File paths for EDA (replace with your actual folders) ---
files = {
    "Finger": "Finger-099/first_EA.csv",
    "Wrist": "Wrist-258/first_EA.csv",
    "Arm":   "Arms-228/first_EA.csv"
}

FS = 15   # Hz (from JSON)
skip_sec = 2

signals, time_axes, snrs = {}, {}, {}

print("\n=== Electrodermal Activity (EDA) ===")

for name, path in files.items():
    df = pd.read_csv(path)

    if "LocalTimestamp" not in df.columns or "EA" not in df.columns:
        print(f"⚠️ Missing columns in {path}")
        continue

    ts = df["LocalTimestamp"].astype(float).values
    sig = df["EA"].astype(float).values

    ts = ts - ts[0]
    mask = ts >= skip_sec
    ts, sig = ts[mask], sig[mask]

    if sig.size == 0:
        print(f"⚠️ {name}: No samples after trimming")
        continue

    # Filter + detrend + normalize
    sig = detrend(sig)
    sig_f = bandpass_filter(sig, fs=FS, low=0.05, high=2)
    sig_f = (sig_f - np.mean(sig_f)) / np.std(sig_f)

    signals[name] = sig_f
    time_axes[name] = ts
    snrs[name] = compute_snr(sig_f)

    print(f"{name}: SNR = {snrs[name]:.2f} dB")

# Align ranges
if signals:
    common_start = max(ts[0] for ts in time_axes.values())
    common_end   = min(ts[-1] for ts in time_axes.values())

    plt.figure(figsize=(12,6))
    for name in signals:
        ts, sig = time_axes[name], signals[name]
        mask = (ts >= common_start) & (ts <= common_end)
        plt.plot(ts[mask], sig[mask], label=f"{name} (SNR={snrs[name]:.2f} dB)")

    plt.title("Electrodermal Activity (EDA) – Finger vs Wrist vs Arm (Aligned, Filtered, Normalized, with SNR)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Conductance (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("EDA_comparison.png", dpi=300)
    plt.show()

print("\n✅ Done! Plot saved as EDA_comparison.png")