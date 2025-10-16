from pylsl import StreamInfo, StreamOutlet
import time

# Create LSL stream info (same name & sourceId used in Oscilloscope JSON)
info = StreamInfo('DataSyncMarker', 'Markers', 1, 0, 'string', '12345')
outlet = StreamOutlet(info)

print("LSL marker stream ready.")
print("Type 'baseline', 'meditation', 'recovery', or 'exit' to send markers.\n")

while True:
    marker = input("Enter marker: ").strip().lower()
    if marker == "exit":
        print("Exiting marker sender.")
        break
    elif marker:
        outlet.push_sample([marker])
        print(f"Marker sent: {marker} at {time.strftime('%H:%M:%S')}")