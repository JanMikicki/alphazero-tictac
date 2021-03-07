# MCTS implementation based on: https://web.archive.org/web/20160308053456/http://mcts.ai/code/python.html
import numpy as np 
from math import *
import random
from copy import deepcopy
import time
import pickle
import matplotlib.pyplot as plt
from decimal import Decimal


"""
This aims to replicate the AlphaZero algorithm for tic-tac-toe/noughts and crosses.
Because the state space is small, it doesn't use neural network.
It seems pretty good after > 20k episodes, but often fails to capitalize of human mistakes.
It still seems a bit worse than vanilla MCTS with rollouts.

Few additional details that real AlphaZero does:
- temperature parameter tau is set to near zero after first 30 moves
  this means it essentially abandons exploration after that
- it rettains tree parameters after descending! (and throws away unnecessary branches)
- adds Dirichlet noise to P(s, a) of (only?) root node
- resigns if root and best child values are below some threshold
- it fiddles with c_pict parameter (for exploration) and c (L2 regularization - only for nets)
"""

def save_dict_to_file(dict_to_save, val_or_pol):
    
    filename = val_or_pol + time.strftime("%Y%m%d-%H%M") + ".pkl"
    a_file = open(filename, "wb")
    pickle.dump(dict_to_save, a_file)
    a_file.close()

def load_dict_from_file(filename):
    
    a_file = open(filename, "rb")
    dict_to_load = pickle.load(a_file)
    a_file.close()
    return dict_to_load

# Value_dict = load_dict_from_file('val20210306-1439.pkl')
# Policy_dict = load_dict_from_file('pol20210306-1439.pkl')
Value_dict = {}
Policy_dict = {}

play_yourself = False       # whether to play against AI or train
n_eps = 100                 # number of training episodes
tau = 0.001                 # temperature parameter - lower means less tree exploration
alpha = 0.001               # learning rate
gamma = 0.99                # discount


class OXOState:
    def __init__(self):
        """ Decide who goes first at the beginning of a game
            (doesn't really matter) and init the board
         """
        self.playerJustMoved = np.random.choice([1, 2])
        self.board = np.zeros((3, 3), dtype=np.uint8)
        
    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
    
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        free_fields = np.where(self.board < 1)
        free_fields = list(zip(*free_fields))
        #import IPython; IPython.embed(); exit()
        return free_fields
    
    def GetValue(self, playerjm, lastplayer):
        """ Get current value for current board and playerjm. 
        """
        board = self.board

        if (board.tobytes(), lastplayer) not in Value_dict:
            #Value_dict[(board.tobytes(), lastplayer)] = np.random.uniform(-1, 1)
            Value_dict[(board.tobytes(), lastplayer)] = 0.0
        
        if playerjm == 2:
            return -Value_dict[(board.tobytes(), lastplayer)]
        else:
            return Value_dict[(board.tobytes(), lastplayer)]

    def GetPolicy(self):
        """ Get the policy distribution for current board and playerjm. 
        """
        if (self.board.tobytes(), self.playerJustMoved) not in Policy_dict:
            x = np.random.rand(9)
            x = np.exp(x)/sum(np.exp(x)) # softmax
            Policy_dict[(self.board.tobytes(), self.playerJustMoved)] = x

        return Policy_dict[(self.board.tobytes(), self.playerJustMoved)]

    def GetResult(self):
        """ Get the game result. Return vals:
            1.0 - someone won
            0.0 - draw
            10  - still ongoing 
        """
        for rowcol in range(3):
            if (self.board[rowcol] == 1).all() \
            or (self.board[:, rowcol] == 1).all():               
                return 1.0

            if (self.board[rowcol] == 2).all() \
            or (self.board[:, rowcol] == 2).all():               
                return 1.0

        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2]:
            if self.board[1, 1] != 0:
                return 1.0
        
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0]:
            if self.board[1, 1] != 0:
                return 1.0

        if len(self.GetMoves()) == 0:
            return 0.0 # draw

        return 10


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.Q = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.policy = state.GetPolicy()
        self.playerJustMoved = state.playerJustMoved
        
    def PUCTSelectChild(self):
        """ Use the PUCT formula to select a child node.
            In short, during tree search we pick actions according to:
            a = argmax(Q + U)
            where U ~ P / 1 + N and Q = wins/visits
            c_puct was chosen based on some paper where they tested different values.
        """
        # s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        c_puct = 3
        
        # Decimal conversion is here because it was getting overflows with high visits and small tau
        s = sorted(self.childNodes, key = lambda c: c.Q + \
            c_puct * self.policy[moves.index(c.move)] * \
                float( (Decimal(sqrt(Decimal(self.visits))) ** Decimal(1/tau)) / (Decimal(c.visits) ** Decimal(1/tau)) ))[-1]

        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result
        self.Q = self.wins / self.visits


def MCTS(rootstate, itermax, verbose = False):
    """ Conduct a Monte Carlo Tree Search for itermax iterations starting from rootstate.
        Return the move policy (probability distribution) from the rootstate.
    """

    rootnode = Node(state = rootstate)
    #print("Starting MCTS...")
    for i in range(itermax):

        node = rootnode
        state = deepcopy(rootstate)

        is_endgame = True if state.GetResult() == 1.0 else False
        
        # Select
        if not is_endgame:
            while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
                node = node.PUCTSelectChild()
                state.DoMove(node.move)
                
        # Expand
        if is_endgame == False and node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m, state) # add child and descend tree

        # Backpropagate
        while node != None:
            # Update nodes with result from POV of node.playerJustMoved
            node.Update(state.GetValue(node.playerJustMoved, state.playerJustMoved)) 
            node = node.parentNode

    # Return the policy based on number of visits
    moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    policy = []
    for m in moves:
        found = False
        for child in rootnode.childNodes:
            if m == child.move:
                policy.append(child.visits/rootnode.visits)
                found = True
                break
        if not found: policy.append(0.0)    # move not possible

    return policy


def play_game_against_random(wins_against_random, moves_against_random):
    # Play one episode against an opponent who's making random moves
    quick_state = OXOState()
    player = np.random.choice([1, 2])
    win = False
    move_cnt = 0

    while (len(quick_state.GetMoves()) != 0):   
        
        if player == 1:
            policy = MCTS(rootstate = quick_state, itermax = 1000, verbose = False)
            m_idx = np.random.choice(9, p=policy)
            moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            m = moves[m_idx]
        else:
            m = random.choice(quick_state.GetMoves())
        
        quick_state.DoMove(m)
        move_cnt += 1
        
        if quick_state.GetResult() == 1.0:
            win = True
            res = 1 if player == 1 else -1
            if res == 1:
                moves_against_random.append(move_cnt)
            wins_against_random.append(res)
            break
        
        player = 3 - player

    if not win:
        wins_against_random.append(0)


def learn(episode_memory, policy_memory, who_won):

    if who_won == 1:        G = 1.0
    elif who_won == 0.0:    G = 0.0
    else:                   G = -1.0
        
    for i, situation in enumerate(reversed(episode_memory)):
        
        board = situation.board

        if i != 0:
            policy = policy_memory[-i]
            org_policy = situation.GetPolicy()

            # Policy could be converted to 2D to make it cooler 
            # and consistent with current move representations...
            new_policy_val = org_policy + alpha * (policy - org_policy)
            new_policy_val = np.clip(new_policy_val, a_min=0.01, a_max=0.99)
            Policy_dict[(board.tobytes(), situation.playerJustMoved)] = new_policy_val

        org_value = situation.GetValue(situation.playerJustMoved, situation.playerJustMoved)

        # Revert back to the actual value in dict
        if situation.playerJustMoved == 2:
            org_value *= -1

        G *= gamma

        new_value = org_value + alpha * (G - org_value)
        Value_dict[(board.tobytes(), situation.playerJustMoved)] = org_value + alpha * (G - org_value)
        

def PlayGames():
    """ Play n_eps games between two equal AI players, and
        update parameters after each episode.
    """
    wins_against_random = []  # just for plotting
    moves_against_random = []

    for episode in range(n_eps):

        if not episode % 100:
            play_game_against_random(wins_against_random, moves_against_random)

            print(f"Episode {episode}")
        
        state = OXOState()
        episode_memory = [deepcopy(state)]
        policy_memory = []       

        #while (len(state.GetMoves()) != 0):
        while (state.GetResult == 10):
            
            policy = MCTS(rootstate = state, itermax = 50, verbose = False)
            m_idx = np.random.choice(9, p=policy)
            moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            m = moves[m_idx]
            policy_memory.append(policy)
            state.DoMove(m)
            episode_memory.append(deepcopy(state))

            # if state.GetResult() == 1.0:
            #     #print(state.playerJustMoved, "won!")
            #     break
        
        if state.GetResult() == 0.0:
            who_won = 0
        else:
            who_won = state.playerJustMoved

        learn(episode_memory, policy_memory, who_won)

    save_dict_to_file(Value_dict, 'val')
    save_dict_to_file(Policy_dict, 'pol')

    plt.plot(wins_against_random, label="wins")
    plt.plot(moves_against_random, label="nr_moves")
    plt.show()


def PlayGameWithHuman():

    state = OXOState()
    
    while (len(state.GetMoves()) != 0):

        policy = MCTS(rootstate = state, itermax = 1000, verbose = False)
        print("policy: ", policy)
        #m_idx = np.random.choice(9, p=policy)
        m_idx = np.argmax(policy)
        moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        m = moves[m_idx]
        print("Chosen Move: ", m, "\n")
        state.DoMove(m)

        print("Real board:")
        print(state.board)

        if state.GetResult() == 1.0:
            print(state.playerJustMoved, "won!")
            break
        elif state.GetResult() == 0.0:
            print("draw")
            break

        print("Pick a move")
        human_m = tuple(int(x.strip()) for x in input().split(','))
        state.DoMove(human_m)

        if state.GetResult() == 1.0:
            print(state.playerJustMoved, "won!")
            break
        elif state.GetResult() == 0.0:
            print("draw")
            break


if __name__ == "__main__":

    if play_yourself:
        PlayGameWithHuman()
    else:
        PlayGames()
    
