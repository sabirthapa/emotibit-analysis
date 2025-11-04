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


def stream_emotibit(serial, name_suffix):
    """Create and stream LSL data from one EmotiBit."""
    global stop_flag

    # initialize board
    params = BrainFlowInputParams()
    params.serial_number = serial
    board = BoardShim(BoardIds.EMOTIBIT_BOARD, params)

    try:
        board.prepare_session()
        board.start_stream()
        print(f"{serial} streaming started...")

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

        print(f"{serial} → LSL outlets ready ({name_suffix})")

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
                    print(f" No data from {serial} for 5 seconds — disconnecting this EmotiBit.")
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
        print(f"Error with {serial}: {e}")

    finally:
        print(f"Stopping {serial}...")
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass
        print(f"{serial} safely disconnected.")


if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()

    # device serials
    serials = [
        # "EM-V6-0000099",
        "EM-V6-0000228",
        # "EM-V6-0000258",
        "EM-V6-0000335",
        # "EM-V6-0000071",
        "EM-V6-0000313"
    ]

    threads = []
    for i, serial in enumerate(serials):
        name = f"EmotiBit_{i+1}"
        t = threading.Thread(target=stream_emotibit, args=(serial, name))
        t.start()
        threads.append(t)

    time.sleep(2)
    print("\nStreaming started for all devices.")
    print("Press 's' (and ENTER) to stop safely.\n")

    while True:
        user_input = input().strip().lower()
        if user_input == "s":
            stop_flag = True
            print("Stopping all devices...")
            break

    # wait for all threads to finish
    for t in threads:
        t.join()

    print("All EmotiBits stopped and released cleanly.")
