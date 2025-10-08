from pylsl import StreamInfo, StreamOutlet
import time

# Create LSL stream info: (name, type, channel_count, nominal_srate, channel_format, source_id)
info = StreamInfo('DataSyncMarker', 'Markers', 1, 0, 'string', '12345')

# Create the outlet for broadcasting
outlet = StreamOutlet(info)

print("LSL marker stream started (DataSyncMarker, source_id=12345)")
print("Sending 'sync' marker every 1 second...")

while True:
    outlet.push_sample(['sync'])
    time.sleep(1)