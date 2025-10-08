from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import time

# Enable logging
BoardShim.enable_dev_board_logger()

# Configure each EmotiBit by serial number
params1 = BrainFlowInputParams()
params1.serial_number = "EM-V6-0000099"

params2 = BrainFlowInputParams()
params2.serial_number = "EM-V6-0000228"


# Initialize all boards
board1 = BoardShim(BoardIds.EMOTIBIT_BOARD, params1)
board2 = BoardShim(BoardIds.EMOTIBIT_BOARD, params2)

# Prepare sessions
board1.prepare_session()
board2.prepare_session()

# Start streaming from all devices
board1.start_stream()
board2.start_stream()

# Collect data
time.sleep(10)

# Get data from each board
data1 = board1.get_board_data()
data2 = board2.get_board_data()

# Stop streaming
board1.stop_stream()
board2.stop_stream()

# Release sessions
board1.release_session()
board2.release_session()
