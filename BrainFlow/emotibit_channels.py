from brainflow.board_shim import BoardShim, BoardIds, BrainFlowPresets
from pprint import pprint

print("\nAUXILIARY (PPG) preset:")
pprint(BoardShim.get_board_descr(BoardIds.EMOTIBIT_BOARD, BrainFlowPresets.AUXILIARY_PRESET))

print("\nANCILLARY (EDA + Temp) preset:")
pprint(BoardShim.get_board_descr(BoardIds.EMOTIBIT_BOARD, BrainFlowPresets.ANCILLARY_PRESET))