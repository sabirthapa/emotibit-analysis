import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, detrend

def bandpass_filter(signal, fs=100, low=0.5, high=5, order=3):
    nyquist = 0.5 * fs
    b, a = butter(order, [low/nyquist, high/nyquist], btype='band')
    return filtfilt(b, a, signal)

def compute_snr(sig):
    power_signal = np.mean(sig**2)
    noise = sig - np.convolve(sig, np.ones(5)/5, mode='same') 
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / power_noise) if power_noise > 0 else -np.inf

files = {
    "Finger": "Finger-099/first_{}.csv",
    "Wrist": "Wrist-258/first_{}.csv",
    "Arm":   "Arms-228/first_{}.csv"
}

channels = {
    "PI": "PPG Red (PI)",
    "PG": "PPG Green (PG)",
    "PR": "PPG Infrared (PR)"
}

FS = 100
skip_sec = 2 

for ch, title in channels.items():
    signals = {}
    time_axes = {}
    snrs = {}

    print(f"\n=== {title} ===")

    for name, template in files.items():
        path = template.format(ch)
        df = pd.read_csv(path)

        if "LocalTimestamp" not in df.columns or ch not in df.columns:
            print(f"Missing columns in {path}")
            continue

        ts = df["LocalTimestamp"].astype(float).values
        sig = df[ch].astype(float).values

        ts = ts - ts[0]
        mask = ts >= skip_sec
        ts, sig = ts[mask], sig[mask]

        if sig.size == 0:
            print(f"{name}-{ch}: No samples after trimming")
            continue

        # Filter + detrend + normalize
        sig = detrend(sig)
        sig_f = bandpass_filter(sig, fs=FS, low=0.5, high=5)
        sig_f = (sig_f - np.mean(sig_f)) / np.std(sig_f)

        signals[name] = sig_f
        time_axes[name] = ts
        snrs[name] = compute_snr(sig_f)

        # Print SNR result
        print(f"{name}: SNR = {snrs[name]:.2f} dB")

    # Align ranges
    if not signals:
        continue
    common_start = max(ts[0] for ts in time_axes.values())
    common_end = min(ts[-1] for ts in time_axes.values())

    plt.figure(figsize=(12,6))
    for name in signals:
        ts, sig = time_axes[name], signals[name]
        mask = (ts >= common_start) & (ts <= common_end)
        plt.plot(ts[mask], sig[mask], 
                 label=f"{name} (SNR={snrs[name]:.2f} dB)")

    plt.title(f"{title} – Finger vs Wrist vs Arm (Aligned, Filtered, Normalized, with SNR)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Signal (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{ch}_comparison.png", dpi=300)
    plt.show()

print("\n Done! Plots saved as PI_comparison.png, PG_comparison.png, PR_comparison.png")
