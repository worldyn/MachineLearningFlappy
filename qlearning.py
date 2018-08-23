from itertools import chain
import pickle

# File to save qtable in
QFILE = 'qvalues.pickle'

# actions
JUMP = 1
NOTHING = 0

# constants
DISCOUNT = 1
ALPHA = 0.7 # learning rate

# Init q-learning array with hashed (dx,dy) states
def init_q():
    q = {}
    # can be behind left side of pipe
    for x in chain(list(range(-40, 140, 10)), list(range(140, 421, 70))):
        # can be under top of lower pipe
        for y in chain(list(range(-300,180,10)), list(range(180,421,60))):
            # the values are for actions 'jump' and 'do nothing'
            q[str(x) + '_' + str(y)] = [0, 0]
    return q

def save_q(qvalues):
    with open(QFILE, 'wb') as f:
        pickle.dump(qvalues, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_q():
    try:
        with open(QFILE, 'rb') as f:
            print("Getting qvalues...")
            return pickle.load(f)
    except:
        print("Initializing new q-table...")
        return init_q()

# map (dx, dy) to the correct 10x10 grid state
def map_to_state(dx, dy):
    dx = int(dx)
    dy = int(dy)
    # most of the game will be dx < 140
    if dx < 140: 
        dx = dx - (dx % 10)
    else:
        dx = dx - (dx % 70)
    # most of the game will be dy < 180
    if dy < 180:
        dy = dy - (dy % 10)
    else:
        dy = dy - (dy % 60)
    return str(dx) + '_' + str(dy)

# returns true for jump action, otherwise false
# if values equal then just do nothing
def select_action(state_hash, qvalues):
    if qvalues[state_hash][JUMP] > qvalues[state_hash][NOTHING]:
        return JUMP
    else:
        return NOTHING

# update according to q-learning algo
# returns true if update ok, otherwise false
def update_qval(qvalues, current_state_hash, new_state_hash,action, reward):
    try: 
        qvalues[current_state_hash][action] = (1-ALPHA) * \
            qvalues[current_state_hash][action] + ALPHA * \
            (reward + DISCOUNT * max(qvalues[new_state_hash]))
        return True
    except KeyError:
        return False
