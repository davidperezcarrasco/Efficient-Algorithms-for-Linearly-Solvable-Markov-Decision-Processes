import numpy as np
from frameworks.lmdp import LMDP
from frameworks.mdp import MDP

class SimpleGrid_LMDP(LMDP):

    def __init__(self, size = 2):
        super().__init__(size * size, 1)
        self.states = []
        
        # construct transition probabilities
        for x in range(size):
            for y in range(size):
                state = x * size + y
                if state < self.n_nonterminal_states:
                    if x > 0:
                        self.P0[state][(x - 1) * size + y] += 1
                    else:
                        self.P0[state][state] += 1
                    if x + 1 < size:
                        self.P0[state][(x + 1) * size + y] += 1
                    else:
                        self.P0[state][state] += 1
                    if y > 0:
                        self.P0[state][x * size + y - 1] += 1
                    else:
                        self.P0[state][state] += 1
                    if y + 1 < size:
                        self.P0[state][x * size + y + 1] += 1
                    else:
                        self.P0[state][state] += 1
                    self.states.append((x, y))

                    self.P0[state][:] /= np.sum(self.P0[state])

        # construct reward function
        self.R[0:self.n_nonterminal_states] = -1

        self.states.append((size - 1, size - 1))

        self.state_to_index = {state: index for index, state in enumerate(self.states)}

class SimpleGrid_MDP(MDP):

    def __init__(self, size = 2):
        super().__init__(size * size, 1, 4)
        self.states = []
        
        # construct transition probabilities
        for x in range(size):
            for y in range(size):
                state = x * size + y
                if state < self.n_nonterminal_states:
                    if x > 0:
                        self.P[state][0][(x - 1) * size + y] = 1
                    else:
                        self.P[state][0][state] = 1
                    if x + 1 < size:
                        self.P[state][1][(x + 1) * size + y] = 1
                    else:
                        self.P[state][1][state] = 1
                    if y > 0:
                        self.P[state][2][x * size + y - 1] = 1
                    else:
                        self.P[state][2][state] = 1
                    if y + 1 < size:
                        self.P[state][3][x * size + y + 1] = 1
                    else:
                        self.P[state][3][state] = 1

        # construct reward function
        self.R[0:self.n_nonterminal_states][:] = -1

        self.states.append((size - 1, size - 1))

        self.state_to_index = {state: index for index, state in enumerate(self.states)}
