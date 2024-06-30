import numpy as np
import frameworks
from scipy.sparse import csr_matrix

class MDP:
    def __init__(self, n_states, n_terminal_states, n_actions, gamma = 1, s0 = 0):
        self.n_states = n_states
        self.n_nonterminal_states = n_states - n_terminal_states
        self.n_actions = n_actions
        self.P = np.zeros((self.n_nonterminal_states, n_actions, n_states))
        self.R = np.zeros((n_states, n_actions)) # Assuming terminal states are at the end of the state space
        self.s0 = s0
        self.gamma = gamma

    def act(self, current_state, action):
        """Transition function."""
        
        next_state = np.random.choice(self.n_states, p=self.P[current_state, action]) 
        reward = self.R[current_state, action]
        terminal = next_state >= self.n_nonterminal_states
        return next_state, reward, terminal
    
    def value_iteration(self, epsilon=1e-10, gamma = None):
        """Value iteration algorithm."""

        gamma = self.gamma if gamma is None else gamma

        #Q = np.concatenate((np.zeros((self.n_nonterminal_states, self.n_actions)), self.R[self.n_nonterminal_states:]), axis=0)
        Q = np.zeros((self.n_states, self.n_actions))
        V_diff = np.arange(self.n_states)
        n_steps = 0

        R = self.R[:self.n_nonterminal_states]
        P = gamma * self.P
        QT = self.R[self.n_nonterminal_states:]

        while max(V_diff) - min(V_diff) > epsilon:
            TQ = R + P @ Q.max(axis=1)
            TQ = np.concatenate((TQ, QT))
            V_diff = TQ.max(axis=1) - Q.max(axis=1)
            Q = TQ
            n_steps += 1

        policy = np.argmax(Q, axis=1)

        return Q, policy, n_steps
    
    def shortest_path_length(self, s = None):
        """Compute the shortest optimal path length from a given state to a terminal state.
        :param s: The starting state. """

        s = self.s0 if s is None else s

        _, policy, _ = self.value_iteration()

        done = s >= self.n_nonterminal_states
        n_steps = 0
        while not done:
            s, _, done = self.act(s, policy[s])
            n_steps += 1
        return n_steps

    def embedding_to_LMDP(self, lmbda = 1, gamma = 1):
        """Embed the MDP into an LMDP."""

        # Compute the value function of the original MDP without discounting
        Q, _, _ = self.value_iteration(gamma=gamma)
        V = Q.max(axis=1)

        # Create the LMDP
        lmdp = frameworks.lmdp.LMDP(self.n_states, self.n_states - self.n_nonterminal_states)
        # Check if all actions from all states are deterministic. Otherwise, the stochastic LMDP embedding will perform better
        is_deterministic = (np.count_nonzero(self.P, axis=2) == np.ones((self.n_nonterminal_states, self.n_actions))).all()

        # Apply the deterministic LMDP embedding
        if is_deterministic:

            lmdp.R = np.sum(self.R, axis = 1)/self.n_actions
            lmdp.P0 = np.sum(self.P, axis = 1)/self.n_actions

            # Update reward function with KL divergence (SPA embedding)
            Z, _ = lmdp.power_iteration(lmbda)
            Pu = lmdp.compute_Pu(Z, sparse=True)
            row_indices = np.repeat(np.arange(Pu.shape[0]), np.diff(Pu.indptr))
            log_ratio = np.log(Pu.data / lmdp.P0[row_indices, Pu.indices])
            product = Pu.data * log_ratio
            R = np.sum(self.R, axis = 1)/self.n_actions + lmbda * np.concatenate((np.bincount(row_indices, weights=product), np.zeros(self.n_states-self.n_nonterminal_states)))

            K_min = 0
            K_max = 1

            #Find the optimal K through ternary search
            while K_max - K_min > 1e-5:
                m1 = K_min + (K_max - K_min) / 3
                lmdp.R = m1 * R
                #lmdp.R[self.n_nonterminal_states:] = R[self.n_nonterminal_states:]
                Z1, _ = lmdp.power_iteration(lmbda)
                mse1 = np.mean(np.square(lmdp.Z_to_V(Z1) - V))
                
                m2 = K_max - (K_max - K_min) / 3
                lmdp.R = m2 * R
                #lmdp.R[self.n_nonterminal_states:] = R[self.n_nonterminal_states:]
                Z2, _ = lmdp.power_iteration(lmbda)
                mse2 = np.mean(np.square(lmdp.Z_to_V(Z2) - V))
                if mse1 > mse2:
                    K_min = m1
                else:
                    K_max = m2

            lmdp.R = K_min * R
            #lmdp.R[self.n_nonterminal_states:] = R[self.n_nonterminal_states:]

        # Apply the non-deterministic LMDP embedding (from Todorov et al. 2009)
        else:
            epsilon = 1e-10
            D = self.P
            # Find columns that contain any non-zero values
            cols_with_nonzero = np.any(D != 0, axis=1)
            unique_next_state_counts = np.unique(np.sum(cols_with_nonzero, axis=1))

            # Perform the vectorized embedding for each unique number of next states
            for next_state_count in unique_next_state_counts:

                source_states = np.where(np.sum(cols_with_nonzero, axis=1) == next_state_count)[0]
                source_states_repeated = np.repeat(source_states, next_state_count)
                next_states = np.where(cols_with_nonzero[source_states])[1]

                # Remove the columns that contain only zeros and keep only possible transitions
                D_count = np.array([D[i][:, cols_with_nonzero[i]] for i in source_states])

                # Substitute 0s in actual possible transitions columns with 'epsilon' and renormalize
                D_count[D_count == 0] = epsilon
                D_count /= D_count.sum(axis=2, keepdims=True)

                # Apply the Pseudo-Inverse method to both square and non-square matrices
                B = self.R[source_states] + np.sum(D_count * np.log(D_count), axis = 2)
                pseudo_inverse_D = np.linalg.pinv(D_count)
                C = np.einsum('ijk,ik->ij', pseudo_inverse_D, B)

                R = np.log(np.sum(np.exp(-C), axis=1))
                M = - R[:, np.newaxis] + C

                # Assign the reward and initial state distribution to the LMDP in the corresponding states
                lmdp.R[source_states] = R
                lmdp.P0[source_states_repeated, next_states] = np.exp(M).flatten()

            lmdp.R[self.n_nonterminal_states:] = np.sum(self.R[self.n_nonterminal_states:], axis = 1)/self.n_actions

        
        embedding_mse = np.mean(np.square(lmdp.Z_to_V(lmdp.power_iteration(lmbda)[0]) - V))
        lmdp.P0 = csr_matrix(lmdp.P0)
        return lmdp, embedding_mse
