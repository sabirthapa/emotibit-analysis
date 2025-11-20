from pylsl import StreamInlet, resolve_streams
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import time

# configs
WINDOW_SIZE = 500          # number of samples to display (≈5 s at 100 Hz)
UPDATE_INTERVAL = 50       # ms between plot updates
DISCOVERY_INTERVAL = 2000  # ms between checking for new streams
NO_DATA_TIMEOUT = 3.0      # seconds of no data before marking as disconnected

# which PPG channel to show from the 3-channel EmotiBit PPG stream
# 0 = first channel, 1 = second, 2 = third
SELECTED_CHANNEL_INDEX = 2  # use channel 3 (often GREEN)
SELECTED_CHANNEL_LABEL = "PPG Ch3"


class EmotiBitMonitor:
    def __init__(self):
        self.inlets = {}          # stream_name -> StreamInlet
        self.data_buffers = {}    # stream_name -> deque of selected channel
        self.serials = {}         # stream_name -> serial string
        self.last_data_time = {}  # stream_name -> last time we got data
        self.is_active = {}       # stream_name -> bool

        self.fig = None
        self.axes = None
        self.lines = {}           # stream_name -> matplotlib Line2D

        self.max_devices = 12
        self.rows = 3
        self.cols = 4

        self.frame_count = 0
        self.discovery_check_interval = DISCOVERY_INTERVAL // UPDATE_INTERVAL

        self.ani = None  # keep animation alive

    # discovery and connection handling
    def discover_new_streams(self):
        """Check for new PPG streams and add them."""
        try:
            all_streams = resolve_streams(wait_time=0.5)
            streams = [s for s in all_streams if s.type() == "PPG"]
        except Exception:
            streams = []

        for stream in streams:
            name = stream.name()
            source_id = stream.source_id()

            # Skip if we already have this stream
            if name in self.inlets:
                continue

            # Extract serial from source_id (e.g., "ppg_EM-V6-0000099")
            serial = source_id.replace("ppg_", "")

            print(f"✓ NEW DEVICE: {name} (Serial: {serial})")

            inlet = StreamInlet(stream, max_buflen=1)

            # single deque for selected channel
            self.data_buffers[name] = deque(maxlen=WINDOW_SIZE)
            self.inlets[name] = inlet
            self.serials[name] = serial
            self.last_data_time[name] = time.time()
            self.is_active[name] = True

            self.add_device_plot(name, serial)

    def check_disconnections(self):
        """Check if any devices have stopped sending data."""
        current_time = time.time()

        for name in list(self.inlets.keys()):
            if self.is_active[name]:
                time_since_data = current_time - self.last_data_time[name]
                if time_since_data > NO_DATA_TIMEOUT:
                    print(
                        f"✗ DISCONNECTED: {self.serials[name]} "
                        f"(no data for {time_since_data:.1f}s)"
                    )
                    self.is_active[name] = False
                    self.mark_inactive(name)

    def mark_inactive(self, name):
        """Gray out a disconnected device's plot."""
        for idx, (stream_name, _) in enumerate(self.serials.items()):
            if stream_name == name:
                ax = self.axes[idx]
                ax.set_facecolor("#f0f0f0")
                ax.set_title(
                    f"{self.serials[name]} [DISCONNECTED]",
                    fontsize=9,
                    fontweight="bold",
                    color="red",
                )
                break

    def reactivate_device(self, name):
        """Reactivate a device that has started sending data again."""
        if not self.is_active[name]:
            print(f"✓ RECONNECTED: {self.serials[name]}")
            self.is_active[name] = True

            for idx, (stream_name, _) in enumerate(self.serials.items()):
                if stream_name == name:
                    ax = self.axes[idx]
                    ax.set_facecolor("white")
                    ax.set_title(
                        f"{self.serials[name]}",
                        fontsize=9,
                        fontweight="bold",
                        color="black",
                    )
                    break

    # plot setup
    def setup_plots(self):
        """Create subplot grid with max capacity."""
        self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=(16, 10))
        self.fig.suptitle(
            f"EmotiBit PPG Monitor – {SELECTED_CHANNEL_LABEL} (single-channel)",
            fontsize=14,
            fontweight="bold",
        )

        self.axes = self.axes.flatten()

        # Hide all subplots initially
        for ax in self.axes:
            ax.set_visible(False)

        plt.tight_layout()

    def add_device_plot(self, name, serial):
        """Add a plot for a new device (one line per device)."""
        device_index = list(self.serials.keys()).index(name)

        if device_index >= len(self.axes):
            print(
                f"⚠ Warning: Maximum devices ({self.max_devices}) reached. "
                f"Cannot display {serial}"
            )
            return

        ax = self.axes[device_index]
        ax.set_visible(True)

        # single line for selected channel
        (line,) = ax.plot([], [], "-", linewidth=0.9, label=SELECTED_CHANNEL_LABEL)

        self.lines[name] = line

        ax.set_xlim(0, WINDOW_SIZE)
        ax.set_ylim(-1, 1)  # will auto-scale later
        ax.set_title(f"{serial}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Samples", fontsize=8)
        ax.set_ylabel("PPG (detrended)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        self.fig.canvas.draw()

    # main animation update 
    def update_plots(self, frame):
        """Update plots and check for new devices."""
        self.frame_count += 1

        # periodically check for new streams and disconnections
        if self.frame_count % self.discovery_check_interval == 0:
            self.discover_new_streams()
            self.check_disconnections()

        # pull new samples for each inlet
        for name, inlet in self.inlets.items():
            samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=100)

            if samples:
                self.last_data_time[name] = time.time()

                if not self.is_active[name]:
                    self.reactivate_device(name)

                buffer = self.data_buffers[name]

                for sample in samples:
                    # pick only selected channel
                    if len(sample) > SELECTED_CHANNEL_INDEX:
                        buffer.append(sample[SELECTED_CHANNEL_INDEX])

        # update each device's line
        for name, line in self.lines.items():
            if not self.is_active[name]:
                continue

            buffer = self.data_buffers[name]
            if len(buffer) == 0:
                continue

            y_raw = np.array(buffer, dtype=float)

            # detrend: subtract mean so we can see pulsatile waveform
            y = y_raw - np.mean(y_raw)

            x = np.arange(len(y))
            line.set_data(x, y)

            # auto-scale y-axis around current data
            y_min, y_max = y.min(), y.max()
            margin = (y_max - y_min) * 0.3 or 1.0

            for idx, (stream_name, _) in enumerate(self.serials.items()):
                if stream_name == name:
                    ax = self.axes[idx]
                    ax.set_xlim(0, WINDOW_SIZE)
                    ax.set_ylim(y_min - margin, y_max + margin)
                    break

        return list(self.lines.values())

    # Run
    def run(self):
        print("=" * 60)
        print("EmotiBit Dynamic Monitor (single-channel PPG)")
        print("=" * 60)
        print("• Devices appear automatically when their PPG stream is found.")
        print("• If a device stops sending data, its panel turns grey + [DISCONNECTED].")
        print("• Close the window to exit.\n")

        self.setup_plots()

        print("Scanning for devices...\n")
        self.discover_new_streams()

        if not self.inlets:
            print("⚠ No PPG streams found yet. I’ll keep scanning...\n")

        # IMPORTANT: keep a reference to the animation
        self.ani = FuncAnimation(
            self.fig,
            self.update_plots,
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False,
        )

        plt.show()
        print("\n✓ Monitor closed.")


if __name__ == "__main__":
    monitor = EmotiBitMonitor()
    monitor.run()