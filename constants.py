CAVE_COLS = 15
CAVE_ROWS = 12
CAVE_CHANNELS = 4
cave_chars_unique = sorted(list(["-","X", "}", "{"]))


MARIO_COLS = 14
MARIO_ROWS = 18
MARIO_CHANNELS = 4
mario_replacements = {'<': 'X', '>': 'X', ']': 'X', "[" : "X", "Q" : "X", "S" : "X"}
mario_chars_unique = sorted(list(["X", "-", "{", "}"]))

SUPERCAT_COLS = 20
SUPERCAT_ROWS = 20
SUPERCAT_CHANNELS = 4
supercat_chars_unique = sorted(list(["-","X", "}", "{"]))

TOMB_COLS = 30
TOMB_ROWS = 15
TOMB_CHANNELS = 4
tomb_chars_unique = sorted(list(["-","X", "}", "{"]))

ICARUS_COLS = 25
ICARUS_ROWS = 16

ZELDA_COLS = 7
ZELDA_ROWS = 11

def get_cols_rows_channels_chars(game):
    return NotImplementedError