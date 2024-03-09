import numpy as np
import random
from constants import CAVE_COLS, CAVE_ROWS, CAVE_CHANNELS, MARIO_COLS, MARIO_ROWS, MARIO_CHANNELS, SUPERCAT_CHANNELS, SUPERCAT_COLS, SUPERCAT_ROWS
from constants import cave_chars_unique, mario_replacements, mario_chars_unique, supercat_chars_unique


def get_dataset(name):
    if name == "cave":
      int2char = dict(enumerate(cave_chars_unique))
      playbale_file_path = './db/cave/playble.txt'
      unplaybale_file_path = './db/cave/unplayble.txt'

    elif name == "mario":
      int2char = dict(enumerate(mario_chars_unique))
      playbale_file_path = './db/mario/playble.txt'
      unplaybale_file_path = './db/mario/unplayble.txt'
    
    elif name == "supercat":
      int2char = dict(enumerate(supercat_chars_unique))
      playbale_file_path = './db/supercat/playble.txt'
      unplaybale_file_path = './db/supercat/unplayble.txt'

    char2int = {ch: ii for ii, ch in int2char.items()}
    num_tiles = len(char2int)

    playble_levels = []
    current_block = []
    with open(playbale_file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if line.startswith(';') or line.startswith('META'):
                if len(current_block)>0:
                    if name == "mario":
                      replaced_block = [mario_replacements.get(char, char) for char in current_block]
                      ncoded_line = [char2int[x] for x in replaced_block]
                    else:
                      ncoded_line = [char2int[x] for x in current_block]
                    playble_levels.append(ncoded_line)
                current_block = []
            elif line:
                current_block.extend(line)

    unplayble_levels = []
    current_block = []
    with open(unplaybale_file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if line.startswith(';') or line.startswith('M'):
                if current_block:
                    if name == "mario":
                      replaced_block = [mario_replacements.get(char, char) for char in current_block]
                      ncoded_line = [char2int[x] for x in replaced_block]
                    else:
                      ncoded_line = [char2int[x] for x in current_block]
                    unplayble_levels.append(ncoded_line)
                current_block = []
            elif line:
                current_block.extend(line)

    # balance the dataset (remove from bigger class)
    if len(unplayble_levels) > len(playble_levels):
      n = len(unplayble_levels) - len(playble_levels)
      for i in range(n):
        index = random.randint(0, len(unplayble_levels)-1)
        unplayble_levels.pop(index)
    elif len(playble_levels) > len(unplayble_levels):
      n = len(playble_levels) - len(unplayble_levels)
      for i in range(n):
        index = random.randint(0, len(playble_levels)-1)
        playble_levels.pop(index)
    print("number of playble levels: ", len(playble_levels))
    playble_levels = np.eye(num_tiles, dtype='uint8')[playble_levels]
    print("number of unplayble levels: ", len(unplayble_levels))
    unplayble_levels = np.eye(num_tiles, dtype='uint8')[unplayble_levels]

    levels = np.concatenate((playble_levels, unplayble_levels))
    labels = np.concatenate((np.ones(len(playble_levels)), np.zeros(len(unplayble_levels))))
    labels = np.array([[1, 0] if label == 0 else [0, 1] for label in labels])

    if name == 'cave':
       levels = levels.reshape(-1,CAVE_COLS,CAVE_ROWS,CAVE_CHANNELS)
    elif name == 'mario':
       levels = levels.reshape(-1,MARIO_COLS, MARIO_ROWS, MARIO_CHANNELS)
    elif name == 'supercat':
       levels = levels.reshape(-1,SUPERCAT_COLS, SUPERCAT_ROWS, SUPERCAT_CHANNELS)
    return levels, labels

def get_level(name, input):
    if name == "cave":
      int2char = dict(enumerate(cave_chars_unique))
    elif name == "mario":
      int2char = dict(enumerate(mario_chars_unique))
    elif name == "supercat":
      int2char = dict(enumerate(supercat_chars_unique))
    char2int = {ch: ii for ii, ch in int2char.items()}
    num_tiles = len(char2int)

    level = []
    with open(input, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if not line.startswith('META'):
              if name == "mario":
                replaced_block = [mario_replacements.get(char, char) for char in line]
                ncoded_line = [char2int[x] for x in replaced_block]
              else:
                ncoded_line = [char2int[x] for x in line]
              level.append(ncoded_line)

    level = np.eye(num_tiles, dtype='uint8')[level]

    if name == 'cave':
       level = level.reshape(1,CAVE_COLS,CAVE_ROWS,CAVE_CHANNELS)
    elif name == 'mario':
       level = level.reshape(1,MARIO_COLS, MARIO_ROWS, MARIO_CHANNELS)
    elif name == 'supercat':
       level = level.reshape(-1,SUPERCAT_COLS, SUPERCAT_ROWS, SUPERCAT_CHANNELS)
    return level

def get_dataset_raw(game):
  if game == "cave":
    playbale_file_path = './db/cave/playble.txt'
    unplaybale_file_path = './db/cave/unplayble.txt'

  elif game == "icarus":
    playbale_file_path = './db/icarus/playble.txt'
    unplaybale_file_path = './db/icarus/unplayble.txt'

  elif game == "mario":
    playbale_file_path = './db/mario/playble.txt'
    unplaybale_file_path = './db/mario/unplayble.txt'

  elif game == "zelda":
    playbale_file_path = './db/zelda/playble.txt'
    unplaybale_file_path = './db/zelda/unplayble.txt'

  playble_levels = []
  unplayble_levels = []

  block = []
  with open(playbale_file_path, 'r') as file:
      for line in file:
          if line.startswith(';') or line.startswith('META'):
              if len(block)>0:
                  playble_levels.append(block)
              block = []
          elif line:
              block.extend(line.rstrip('\n'))

  block = []
  with open(unplaybale_file_path, 'r') as file:
      for line in file:
          if line.startswith(';') or line.startswith('META'):
              if len(block)>0:
                  unplayble_levels.append(block)
              block = []
          elif line:
              block.extend(line.rstrip('\n'))

  # balance the dataset (remove from bigger class)
  if len(unplayble_levels) > len(playble_levels):
    n = len(unplayble_levels) - len(playble_levels)
    for i in range(n):
      index = random.randint(0, len(unplayble_levels)-1)
      unplayble_levels.pop(index)
  elif len(playble_levels) > len(unplayble_levels):
    n = len(playble_levels) - len(unplayble_levels)
    for i in range(n):
      index = random.randint(0, len(playble_levels)-1)
      playble_levels.pop(index)

  if game == 'cave':
      playble_levels = np.array(playble_levels).reshape(-1,CAVE_COLS,CAVE_ROWS)
      unplayble_levels = np.array(unplayble_levels).reshape(-1,CAVE_COLS,CAVE_ROWS)
  elif game == 'mario':
      playble_levels = np.array(playble_levels).reshape(-1,MARIO_COLS, MARIO_ROWS)
      unplayble_levels = np.array(unplayble_levels).reshape(-1,MARIO_COLS, MARIO_ROWS)
  elif game == 'supercat':
       playble_levels = np.array(playble_levels).reshape(-1,SUPERCAT_COLS, SUPERCAT_ROWS, SUPERCAT_CHANNELS)
       unplayble_levels = np.array(unplayble_levels).reshape(-1,SUPERCAT_COLS, SUPERCAT_ROWS, SUPERCAT_CHANNELS)
  return (playble_levels, unplayble_levels)

def get_level_raw(name, input):
    level = []
    with open(input, 'r') as file:
        for line in file:
            if not line.startswith('META'):
              level.extend(line.rstrip('\n'))
    if name == 'cave':
       level = np.array(level).reshape(-1,CAVE_COLS,CAVE_ROWS)
    elif name == 'mario':
       level = np.array(level).reshape(-1,MARIO_COLS, MARIO_ROWS)
    elif name == 'supercat':
       level = np.array(level).reshape(-1,SUPERCAT_COLS, SUPERCAT_ROWS)
    return level

def get_dataset_flat(name, remove_empty = True):
    if name == "cave":
      int2char = dict(enumerate(cave_chars_unique))
      playbale_file_path = './db/cave/playble.txt'
      unplaybale_file_path = './db/cave/unplayble.txt'
    elif name == "mario":
      int2char = dict(enumerate(mario_chars_unique))
      playbale_file_path = './db/mario/playble.txt'
      unplaybale_file_path = './db/mario/unplayble.txt'

    char2int = {ch: ii for ii, ch in int2char.items()}
    num_tiles = len(char2int)

    playble_levels = []
    current_block = []
    with open(playbale_file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if line.startswith(';') or line.startswith('META'):
                if len(current_block)>0:
                    ncoded_line = [char2int[x] for x in current_block]
                    playble_levels.append(ncoded_line)
                current_block = []
            elif line:
                current_block.extend(line)

    unplayble_levels = []
    current_block = []
    with open(unplaybale_file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if line.startswith(';'):
                if current_block:
                    if name == "mario":
                      replaced_block = [mario_replacements.get(char, char) for char in line]
                      ncoded_line = [char2int[x] for x in replaced_block]
                    else:
                      ncoded_line = [char2int[x] for x in current_block]
                    unplayble_levels.append(ncoded_line)
                current_block = []
            elif line:
                current_block.extend(line)

    # balance the dataset (remove from bigger class)
    if len(unplayble_levels) > len(playble_levels):
      n = len(unplayble_levels) - len(playble_levels)
      for i in range(n):
        index = random.randint(0, len(unplayble_levels)-1)
        unplayble_levels.pop(index)
    elif len(playble_levels) > len(unplayble_levels):
      n = len(playble_levels) - len(unplayble_levels)
      for i in range(n):
        index = random.randint(0, len(playble_levels)-1)
        playble_levels.pop(index)

    levels = np.concatenate((playble_levels, unplayble_levels))
    labels = np.concatenate((np.ones(len(playble_levels)), np.zeros(len(unplayble_levels))))

    if name == 'cave':
       levels = levels.reshape(-1,CAVE_COLS,CAVE_ROWS)
    elif name == 'mario':
       levels = levels.reshape(-1,MARIO_COLS, MARIO_ROWS)
    elif name == 'supercat':
       levels = levels.reshape(-1,SUPERCAT_COLS, SUPERCAT_ROWS)
    return levels, labels

def get_level_flat(name, input):
    if name == "cave":
      int2char = dict(enumerate(cave_chars_unique))
    elif name == "mario":
      int2char = dict(enumerate(mario_chars_unique))
    elif name == "supercat":
      int2char = dict(enumerate(supercat_chars_unique))
    char2int = {ch: ii for ii, ch in int2char.items()}
    num_tiles = len(char2int)

    level = []
    with open(input, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if not line.startswith('META'):
              if name == "mario":
                replaced_block = [mario_replacements.get(char, char) for char in line]
                ncoded_line = [char2int[x] for x in replaced_block]
              else:
                ncoded_line = [char2int[x] for x in line]
              level.append(ncoded_line)

    return np.array(level)