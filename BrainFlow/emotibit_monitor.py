from pylsl import StreamInlet, resolve_streams
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import time

# Configuration
WINDOW_SIZE = 500  # number of samples to display (5 seconds at 100Hz)
UPDATE_INTERVAL = 50  # milliseconds between plot updates
DISCOVERY_INTERVAL = 2000  # milliseconds between checking for new streams
NO_DATA_TIMEOUT = 3.0  # seconds of no data before marking as disconnected

class EmotiBitMonitor:
    def __init__(self):
        self.inlets = {}
        self.data_buffers = {}
        self.serials = {}
        self.last_data_time = {}
        self.is_active = {}
        
        self.fig = None
        self.axes = None
        self.lines = {}
        
        self.max_devices = 12
        self.rows = 3
        self.cols = 4
        
        self.frame_count = 0
        self.discovery_check_interval = DISCOVERY_INTERVAL // UPDATE_INTERVAL
        
    def discover_new_streams(self):
        """Check for new PPG streams and add them."""
        # resolve_streams returns all streams, we filter for PPG
        try:
            all_streams = resolve_streams(wait_time=0.5)
            streams = [s for s in all_streams if s.type() == 'PPG']
        except:
            streams = []
        
        for stream in streams:
            name = stream.name()
            source_id = stream.source_id()
            
            # Skip if we already have this stream
            if name in self.inlets:
                continue
            
            # Extract serial from source_id (format: "ppg_EM-V6-0000099")
            serial = source_id.replace("ppg_", "")
            
            print(f"✓ NEW DEVICE: {name} (Serial: {serial})")
            
            # Create inlet
            inlet = StreamInlet(stream, max_buflen=1)
            
            # Initialize data buffer for 3 channels
            self.data_buffers[name] = {
                'ch1': deque(maxlen=WINDOW_SIZE),
                'ch2': deque(maxlen=WINDOW_SIZE),
                'ch3': deque(maxlen=WINDOW_SIZE),
            }
            
            self.inlets[name] = inlet
            self.serials[name] = serial
            self.last_data_time[name] = time.time()
            self.is_active[name] = True
            
            # Add plot for this device
            self.add_device_plot(name, serial)
    
    def check_disconnections(self):
        """Check if any devices have stopped sending data."""
        current_time = time.time()
        
        for name in list(self.inlets.keys()):
            if self.is_active[name]:
                time_since_data = current_time - self.last_data_time[name]
                
                if time_since_data > NO_DATA_TIMEOUT:
                    print(f"✗ DISCONNECTED: {self.serials[name]} (no data for {time_since_data:.1f}s)")
                    self.is_active[name] = False
                    self.mark_inactive(name)
    
    def mark_inactive(self, name):
        """Gray out a disconnected device's plot."""
        for idx, (stream_name, _) in enumerate(self.serials.items()):
            if stream_name == name:
                ax = self.axes[idx]
                ax.set_facecolor('#f0f0f0')
                ax.set_title(f"{self.serials[name]} [DISCONNECTED]", 
                           fontsize=9, fontweight='bold', color='red')
                break
    
    def reactivate_device(self, name):
        """Reactivate a device that has started sending data again."""
        if not self.is_active[name]:
            print(f"✓ RECONNECTED: {self.serials[name]}")
            self.is_active[name] = True
            
            for idx, (stream_name, _) in enumerate(self.serials.items()):
                if stream_name == name:
                    ax = self.axes[idx]
                    ax.set_facecolor('white')
                    ax.set_title(f"{self.serials[name]}", 
                               fontsize=9, fontweight='bold', color='black')
                    break
    
    def setup_plots(self):
        """Create subplot grid with max capacity."""
        self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=(16, 10))
        self.fig.suptitle('EmotiBit PPG Monitor - Dynamic Real-time Visualization', 
                         fontsize=14, fontweight='bold')
        
        # Flatten axes
        self.axes = self.axes.flatten()
        
        # Hide all subplots initially
        for ax in self.axes:
            ax.set_visible(False)
        
        plt.tight_layout()
    
    def add_device_plot(self, name, serial):
        """Add a plot for a new device."""
        # Find the next available subplot
        device_index = list(self.serials.keys()).index(name)
        
        if device_index >= len(self.axes):
            print(f"⚠ Warning: Maximum devices ({self.max_devices}) reached. Cannot display {serial}")
            return
        
        ax = self.axes[device_index]
        ax.set_visible(True)
        
        # Create lines for 3 PPG channels
        line1, = ax.plot([], [], 'r-', linewidth=0.8, label='Ch1', alpha=0.7)
        line2, = ax.plot([], [], 'g-', linewidth=0.8, label='Ch2', alpha=0.7)
        line3, = ax.plot([], [], 'b-', linewidth=0.8, label='Ch3', alpha=0.7)
        
        self.lines[name] = [line1, line2, line3]
        
        ax.set_xlim(0, WINDOW_SIZE)
        ax.set_ylim(-5000, 5000)
        ax.set_title(f"{serial}", fontsize=9, fontweight='bold')
        ax.set_xlabel('Samples', fontsize=8)
        ax.set_ylabel('PPG', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()
    
    def update_plots(self, frame):
        """Update all plots with new data and check for new devices."""
        self.frame_count += 1
        
        # Periodically check for new streams and disconnections
        if self.frame_count % self.discovery_check_interval == 0:
            self.discover_new_streams()
            self.check_disconnections()
        
        # Update data for all devices
        for name, inlet in self.inlets.items():
            # Pull all available samples
            samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=100)
            
            if samples:
                self.last_data_time[name] = time.time()
                
                # Reactivate if was inactive
                if not self.is_active[name]:
                    self.reactivate_device(name)
                
                buffer = self.data_buffers[name]
                
                # Add new samples to buffers
                for sample in samples:
                    buffer['ch1'].append(sample[0])
                    buffer['ch2'].append(sample[1])
                    buffer['ch3'].append(sample[2])
        
        # Update all line plots
        for name, lines in self.lines.items():
            if not self.is_active[name]:
                continue
                
            buffer = self.data_buffers[name]
            
            if len(buffer['ch1']) > 0:
                x = np.arange(len(buffer['ch1']))
                
                lines[0].set_data(x, list(buffer['ch1']))
                lines[1].set_data(x, list(buffer['ch2']))
                lines[2].set_data(x, list(buffer['ch3']))
                
                # Auto-scale y-axis based on current data
                all_data = list(buffer['ch1']) + list(buffer['ch2']) + list(buffer['ch3'])
                if all_data:
                    y_min, y_max = min(all_data), max(all_data)
                    margin = (y_max - y_min) * 0.1 or 100
                    
                    # Get the axis for this stream
                    for idx, (stream_name, _) in enumerate(self.serials.items()):
                        if stream_name == name:
                            self.axes[idx].set_ylim(y_min - margin, y_max + margin)
                            break
        
        return [line for lines in self.lines.values() for line in lines]
    
    def run(self):
        """Start the monitoring interface."""
        print("="*60)
        print("EmotiBit Dynamic Monitor")
        print("="*60)
        print("Starting monitor...")
        print("• Devices will appear automatically when connected")
        print("• Disconnected devices will be marked in red")
        print("• Close the plot window to exit")
        print("="*60 + "\n")
        
        self.setup_plots()
        
        # Initial discovery
        print("Scanning for devices...\n")
        self.discover_new_streams()
        
        if not self.inlets:
            print("⚠ No devices found yet. Monitor will keep scanning...")
            print("   Start your streaming script if you haven't already.\n")
        
        # Create animation
        ani = FuncAnimation(
            self.fig, 
            self.update_plots, 
            interval=UPDATE_INTERVAL,
            blit=False,  # Changed to False for dynamic updates
            cache_frame_data=False
        )
        
        plt.show()
        
        print("\n✓ Monitor closed.")


if __name__ == "__main__":
    monitor = EmotiBitMonitor()
    monitor.run()
