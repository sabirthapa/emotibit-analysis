from pylsl import StreamInfo, StreamOutlet
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowPresets,
)
import time
import threading

# global stop flag shared by all threads
stop_flag = False


def stream_emotibit(serial, name_suffix, init_delay=0):
    """Create and stream LSL data from one EmotiBit."""
    global stop_flag

    # stagger initialization
    if init_delay > 0:
        time.sleep(init_delay)

    # initialize board
    params = BrainFlowInputParams()
    params.serial_number = serial
    board = BoardShim(BoardIds.EMOTIBIT_BOARD, params)

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries and not stop_flag:
        try:
            print(f"[{serial}] Attempting connection (attempt {retry_count + 1}/{max_retries})...")
            board.prepare_session()
            board.start_stream()
            print(f"✓ [{serial}] streaming started!")
            break
        except Exception as e:
            retry_count += 1
            print(f"✗ [{serial}] Connection failed: {e}")
            if retry_count < max_retries:
                wait_time = 2 * retry_count  # exponential backoff
                print(f"  [{serial}] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                try:
                    board.release_session()
                except:
                    pass
            else:
                print(f"✗ [{serial}] Failed after {max_retries} attempts. Skipping this device.")
                return

    if stop_flag:
        return

    try:
        # create LSL outlets
        ppg_info = StreamInfo(
            f"PPG_{name_suffix}", "PPG", 3, 100, "float32", f"ppg_{serial}"
        )
        eda_info = StreamInfo(
            f"EDA_{name_suffix}", "EDA", 1, 15, "float32", f"eda_{serial}"
        )
        temp_info = StreamInfo(
            f"TEMP_{name_suffix}", "TEMP", 1, 15, "float32", f"temp_{serial}"
        )

        ppg_outlet = StreamOutlet(ppg_info)
        eda_outlet = StreamOutlet(eda_info)
        temp_outlet = StreamOutlet(temp_info)

        print(f"✓ [{serial}] LSL outlets ready ({name_suffix})")

        # track empty polls
        empty_read_count = 0
        
        # main streaming loop
        while not stop_flag:
            aux_data = board.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)
            anc_data = board.get_board_data(preset=BrainFlowPresets.ANCILLARY_PRESET)

            # if no data from EmotiBit
            if aux_data.shape[1] == 0 and anc_data.shape[1] == 0:
                empty_read_count += 1
                # if no data for ~5 seconds (0.001s * 5000 loops)
                if empty_read_count > 5000:
                    print(f"⚠ [{serial}] No data for 5 seconds — disconnecting.")
                    break
                time.sleep(0.001)
                continue
            else:
                empty_read_count = 0
            
            # push PPG
            if aux_data.shape[1] > 0:
                for i in range(aux_data.shape[1]):
                    ppg_outlet.push_sample(aux_data[1:4, i].tolist())

            # push EDA + Temp
            if anc_data.shape[1] > 0:
                for i in range(anc_data.shape[1]):
                    eda_outlet.push_sample([anc_data[1, i]])
                    temp_outlet.push_sample([anc_data[2, i]])

            time.sleep(0.001)

    except Exception as e:
        print(f"✗ [{serial}] Error during streaming: {e}")

    finally:
        print(f"⊗ [{serial}] Stopping...")
        try:
            board.stop_stream()
            board.release_session()
        except Exception as e:
            print(f"  [{serial}] Error during cleanup: {e}")
        print(f"⊗ [{serial}] Disconnected.")


if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()

    # device serials
    serials = [
        "EM-V6-0000099",
        "EM-V6-0000228",
        "EM-V6-0000258",
        "EM-V6-0000335",
        "EM-V6-0000071",
        "EM-V6-0000313",
        "EM-V6-0000324",
        "EM-V6-0000135",
        "EM-V6-0000304",
        "EM-V6-0000146",
        "EM-V6-0000038",
        "EM-V6-0000274",
    ]

    print(f"Starting connection to {len(serials)} EmotiBit devices...")
    print("Devices will be initialized with staggered delays to prevent conflicts.\n")

    threads = []
    stagger_delay = 1.5  # seconds between each device initialization
    
    for i, serial in enumerate(serials):
        name = f"EmotiBit_{i+1}"
        # Each device starts with increasing delay
        t = threading.Thread(
            target=stream_emotibit, 
            args=(serial, name, i * stagger_delay)
        )
        t.start()
        threads.append(t)

    # Wait for all initialization attempts to complete
    total_init_time = len(serials) * stagger_delay + 5
    print(f"Waiting {total_init_time:.0f} seconds for all devices to initialize...\n")
    time.sleep(total_init_time)

    print("\n" + "="*60)
    print("Initialization complete!")
    print("Press 's' (and ENTER) to stop all devices safely.")
    print("="*60 + "\n")

    while True:
        user_input = input().strip().lower()
        if user_input == "s":
            stop_flag = True
            print("\n⊗ Stopping all devices...")
            break

    # wait for all threads to finish
    for t in threads:
        t.join()

    print("\n✓ All EmotiBits stopped and released cleanly.")