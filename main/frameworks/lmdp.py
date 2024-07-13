import numpy as np
from frameworks.mdp import MDP
from scipy.sparse import csr_matrix, isspmatrix_csr


class LMDP:
    def __init__(self, n_states, n_terminal_states, lmbda = 1, s0 = 0):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.P0 = np.zeros((self.n_nonterminal_states, n_states))
        self.R = np.zeros(n_states) # Assuming terminal states are at the end of the state space
        self.s0 = s0
        self.lmbda = lmbda

    def act(self, current_state, P):
        """Transition function."""

        # Check if the transition matrix is sparse 
        if isspmatrix_csr(P):
            next_state = np.random.choice(P[current_state].indices, p=P[current_state].data) # Using sparse matrix
        else:
            next_state = np.random.choice(self.n_states, p=P[current_state])
        reward = self.R[next_state]
        terminal = next_state >= self.n_nonterminal_states
        return next_state, reward, terminal
    
    def power_iteration(self, lmbda = None, epsilon = 1e-10):
        """Power iteration algorithm to compute the optimal Z function."""

        lmbda = self.lmbda if lmbda is None else lmbda

        P0 = self.P0 if isspmatrix_csr(self.P0) else csr_matrix(self.P0)
        
        #Z = np.concatenate((np.ones(self.n_nonterminal_states), np.exp(self.R[self.n_nonterminal_states:] / lmbda)))
        Z = np.ones(self.n_states)
        V_diff = np.arange(self.n_states)
        n_steps = 0
        
        G = csr_matrix(np.diag(np.exp(self.R[:self.n_nonterminal_states] / lmbda)))
        ZT = np.exp(self.R[self.n_nonterminal_states:] / lmbda)

        while max(V_diff) - min(V_diff) > epsilon:
            TZ = G @ P0 @ Z
            TZ = np.concatenate((TZ, ZT))
            V_diff = self.Z_to_V(TZ) - self.Z_to_V(Z)
            Z = TZ
            n_steps += 1

        return Z, n_steps
    
    
    def compute_Pu(self, Z, sparse = True):
        if sparse:
            P0 = csr_matrix(self.P0) if not isspmatrix_csr(self.P0) else self.P0
            Pu = P0.multiply(Z) # Element-wise multiplication of P0 and Z
            # Normalize each row of the matrix
            row_sums = Pu.sum(axis=1)
            Pu = Pu.multiply(csr_matrix(1.0 / row_sums))

        else:
            P0 = self.P0.toarray() if isspmatrix_csr(self.P0) else self.P0
            Pu = self.P0 * Z  # Element-wise multiplication of P0 and Z
            # Normalize each row of the matrix
            row_sums = Pu.sum(axis=1, keepdims=True)
            Pu /= row_sums
        return Pu
    
    def Z_to_V(self, Z, lmbda = None):
        lmbda = self.lmbda if lmbda is None else lmbda
        V = lmbda * np.log(Z)
        return V
  
    def embedding_to_MDP(self, lmbda = None):
        """Embed the LMDP into an MDP."""

        lmbda = self.lmbda if lmbda is None else lmbda
        
        # Extract the number of actions from nonzero transition probabilities
        P0 = self.P0.toarray() if isspmatrix_csr(self.P0) else self.P0
        n_actions = np.max((P0 > 0).sum(axis=1))
        mdp = MDP(self.n_states, self.n_states - self.n_nonterminal_states, n_actions)
        Z_opt, _ = self.power_iteration(lmbda)
        Pu = self.compute_Pu(Z_opt)

        # Compute the reward function
        row_indices = np.repeat(np.arange(Pu.shape[0]), np.diff(Pu.indptr))
        log_ratio = np.log(Pu.data / P0[row_indices, Pu.indices])
        product = Pu.data * log_ratio
        mdp.R = self.R - lmbda * np.concatenate((np.bincount(row_indices, weights=product), np.zeros(self.n_states-self.n_nonterminal_states)))
        mdp.R = np.broadcast_to(mdp.R.reshape(-1, 1), (self.n_states, n_actions))
        
        # Compute the transition probabilities
        n_next_states_per_row = np.diff(Pu.indptr)
        n_next_states = np.unique(n_next_states_per_row)

        # Iterate through all possible transition dimensionalities to avoid heterogeneous matrices
        for next_states in n_next_states:
            source_states = np.where(n_next_states_per_row == next_states)[0]
            source_states_repeated = np.repeat(source_states, next_states)
            indices = Pu[source_states].indices.reshape(-1, next_states)

            for a in range(mdp.n_actions):
                rolled_indices = np.roll(indices, -a, axis=1).flatten()
                mdp.P[source_states_repeated, a, rolled_indices] = Pu[source_states].data

        # Compute the embedding error
        V_lmdp = self.Z_to_V(Z_opt)
        Q, _, _ = mdp.value_iteration(gamma=1)
        V_mdp = Q.max(axis=1)
        embedding_mse = np.mean(np.square(V_lmdp - V_mdp))

        return mdp, embedding_mse
    
    def shortest_path_length(self, s=None):
        """Compute the shortest optimal path length from a given state to a terminal state.
        :param s: The starting state. """

        s = self.s0 if s is None else s

        Pu = self.compute_Pu(self.power_iteration()[0])

        done = s >= self.n_nonterminal_states
        n_steps = 0
        while not done:
            s, _, done = self.act(s, Pu)
            n_steps += 1
        return n_steps
