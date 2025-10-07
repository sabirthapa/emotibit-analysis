import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, detrend

# --- Low-pass filter for Temperature (<0.5 Hz) ---
def lowpass_filter(signal, fs=7.5, cutoff=0.5, order=3):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff/nyquist, btype='low')
    return filtfilt(b, a, signal)

# --- SNR calculation ---
def compute_snr(sig):
    power_signal = np.mean(sig**2)
    noise = sig - np.convolve(sig, np.ones(5)/5, mode='same')
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / power_noise) if power_noise > 0 else -np.inf

# --- File paths ---
files = {
    "Finger": "Finger-099/first_T1.csv",
    "Wrist":  "Wrist-258/first_T1.csv",
    "Arm":    "Arms-228/first_T1.csv"
}

FS = 7.5   # Hz for temperature channel
skip_sec = 2

signals, time_axes, snrs = {}, {}, {}

print("\n=== Skin Temperature (T1) ===")

for name, path in files.items():
    df = pd.read_csv(path)

    if "LocalTimestamp" not in df.columns or "T1" not in df.columns:
        print(f"⚠️ Missing columns in {path}")
        continue

    ts = df["LocalTimestamp"].astype(float).values
    sig = df["T1"].astype(float).values

    ts = ts - ts[0]
    mask = ts >= skip_sec
    ts, sig = ts[mask], sig[mask]

    if sig.size == 0:
        print(f"⚠️ {name}-T1: No samples after trimming")
        continue

    # Filter + detrend + normalize
    sig = detrend(sig)
    sig_f = lowpass_filter(sig, fs=FS, cutoff=0.5)
    sig_f = (sig_f - np.mean(sig_f)) / np.std(sig_f)

    signals[name] = sig_f
    time_axes[name] = ts
    snrs[name] = compute_snr(sig_f)

    print(f"{name}: SNR = {snrs[name]:.2f} dB")

# --- Align common time window ---
if signals:
    common_start = max(ts[0] for ts in time_axes.values())
    common_end   = min(ts[-1] for ts in time_axes.values())

    plt.figure(figsize=(12,6))
    for name in signals:
        ts, sig = time_axes[name], signals[name]
        mask = (ts >= common_start) & (ts <= common_end)
        plt.plot(ts[mask], sig[mask], label=f"{name} (SNR={snrs[name]:.2f} dB)")

    plt.title("Skin Temperature (T1) – Finger vs Wrist vs Arm (Aligned, Filtered, Normalized, with SNR)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Temperature (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("T1_comparison.png", dpi=300)
    plt.show()

print("\n✅ Done! Plot saved as T1_comparison.png")