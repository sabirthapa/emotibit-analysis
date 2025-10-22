import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

def main():
    # enable detailed BrainFlow logs
    BoardShim.enable_dev_board_logger()

    # emotibit serial numbers
    SERIAL_1 = "EM-V6-0000099"
    SERIAL_2 = "EM-V6-0000228"

    # initialize connection parameters
    params1 = BrainFlowInputParams()
    params1.serial_number = SERIAL_1
    params2 = BrainFlowInputParams()
    params2.serial_number = SERIAL_2

    board_id = BoardIds.EMOTIBIT_BOARD.value
    presets = BoardShim.get_board_presets(board_id)
    print("Available presets:", presets)

    # initialize both EmotiBits
    board1 = BoardShim(board_id, params1)
    board2 = BoardShim(board_id, params2)
    print("🔌 Preparing EmotiBits...")
    board1.prepare_session()
    board2.prepare_session()

    # start streaming for both
    print("Recording started! Press ENTER anytime to stop...")
    board1.start_stream()
    board2.start_stream()

    try:
        input()  # waits until you press ENTER manually
    finally:
        print("Stopping streams...")

        # fetch data from both boards
        data_default1 = board1.get_board_data(preset=BrainFlowPresets.DEFAULT_PRESET)
        data_aux1 = board1.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)
        data_anc1 = board1.get_board_data(preset=BrainFlowPresets.ANCILLARY_PRESET)

        data_default2 = board2.get_board_data(preset=BrainFlowPresets.DEFAULT_PRESET)
        data_aux2 = board2.get_board_data(preset=BrainFlowPresets.AUXILIARY_PRESET)
        data_anc2 = board2.get_board_data(preset=BrainFlowPresets.ANCILLARY_PRESET)

        # stop and release sessions
        board1.stop_stream()
        board2.stop_stream()
        board1.release_session()
        board2.release_session()

        # save all presets to CSV
        print("Saving data to CSV files...")
        DataFilter.write_file(data_default1, "emotibit_099_default.csv", "w")
        DataFilter.write_file(data_aux1, "emotibit_099_aux.csv", "w")
        DataFilter.write_file(data_anc1, "emotibit_099_anc.csv", "w")

        DataFilter.write_file(data_default2, "emotibit_228_default.csv", "w")
        DataFilter.write_file(data_aux2, "emotibit_228_aux.csv", "w")
        DataFilter.write_file(data_anc2, "emotibit_228_anc.csv", "w")

        print("Done! Files saved:")
        print(" emotibit_099_default.csv / aux.csv / anc.csv")
        print(" emotibit_228_default.csv / aux.csv / anc.csv")

if __name__ == "__main__":
    main()