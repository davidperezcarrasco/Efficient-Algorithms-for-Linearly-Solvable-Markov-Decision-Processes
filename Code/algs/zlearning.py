import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import time

#-------------------------------
# Z-Learning implementation
# ------------------------------

class ZLearning:
    """
    Implements Z-learning algorithm
    """
    def __init__(self, lmdp, lmbda=1, c = 10000, reset_randomness = 0.0, naive = False):
        self.lmdp = lmdp
        self.lmbda = lmbda
        self.learning_rate = 1
        self.c = c
        self.n_episodes = 0
        self.Z = np.ones(lmdp.n_states)
        self.Z[self.lmdp.n_nonterminal_states:] = np.exp(self.lmdp.R[self.lmdp.n_nonterminal_states:] / lmbda)
        self.P0 = self.lmdp.P0
        self.Pu = self.lmdp.P0
        self.reset_randomness = reset_randomness
        self.state = self.lmdp.s0
        self.r = 0
        self.episode_end = False
        self.naive = naive


    def get_Z(self, r, x, y):
        """
        :param r: reward of current state
        :param x: current state (in index format)
        :param y: next state (in index format)
        :return: Delta update
        """

        Gz = self.Z[y] if self.naive else self.lmdp.P0[x].dot(self.Z)

        zjk = np.exp(r / self.lmbda) * Gz

        return zjk - self.Z[x]

    def step(self):

        # Sample next state
        next_state, _ , self.episode_end = self.lmdp.act(self.state, self.Pu) if not self.naive else self.lmdp.act(self.state, self.P0)

        # Get Delta
        self.r = self.lmdp.R[self.state]
        delta = self.get_Z(self.r, self.state, next_state)

        # Update Z
        self.Z[self.state] += self.learning_rate * delta

        # Compute Pu
        if not self.naive:
            self.Pu = self.lmdp.compute_Pu(self.Z)

        # Update state
        self.state = next_state

        if self.episode_end:
            self.r += self.lmdp.R[self.state]
            self.n_episodes += 1
            self.learning_rate = self.c / (self.c + self.n_episodes)
            self.state = np.random.choice(self.lmdp.n_nonterminal_states) if np.random.rand() < self.reset_randomness else self.lmdp.s0

def Zlearning_training(zlearning: ZLearning, n_steps = int(5e5), V=None):
    tt = 0
    l0 = 0
    z_lengths = []
    cumulative_reward = 0
    z_rewards = np.zeros(n_steps)
    z_throughputs = np.zeros(n_steps)

    V_error = np.zeros((n_steps))
    start_time = time.time()
    while tt < n_steps:
        zlearning.step()
        if tt < len(V_error) and V is not None:
            V_est = zlearning.lmdp.Z_to_V(zlearning.Z)
            V_error[tt] = np.mean(np.square(V_est - V))
        cumulative_reward += zlearning.r
        tt += 1

        if zlearning.episode_end:
            z_lengths.append(tt-l0)
            z_rewards[l0:tt] = cumulative_reward
            z_throughputs[l0:tt] = -1/(cumulative_reward)
            l0 = tt
            cumulative_reward = 0

        if tt % 10000 == 0:

            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / tt) * n_steps
            estimated_remaining_time = estimated_total_time - elapsed_time

            print(f"Step: {tt}/{n_steps}, Time: {elapsed_time/60:.2f}m, ETA: {estimated_remaining_time/60:.2f}m")


    if l0 != tt: 
        z_throughputs[l0:tt] = -1/(cumulative_reward)
        z_rewards[l0:tt] = cumulative_reward

    zlearning.Pu = zlearning.lmdp.compute_Pu(zlearning.Z)

    return zlearning.Z, V_error, z_throughputs, z_rewards
