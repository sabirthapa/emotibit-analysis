from pylsl import StreamInfo, StreamOutlet
import time

# create a marker stream 
info = StreamInfo("MarkerStream", "Markers", 1, 0, "string", "marker_stream_id")
outlet = StreamOutlet(info)
print("✅ Marker stream ready. Press keys to send markers:")

# send markers manually
try:
    while True:
        marker = input("Enter marker (or 'q' to quit): ")
        if marker.lower() == "q":
            break
        outlet.push_sample([marker])
        print(f"📍 Sent marker: {marker}")

except KeyboardInterrupt:
    print("Stopped marker stream.")