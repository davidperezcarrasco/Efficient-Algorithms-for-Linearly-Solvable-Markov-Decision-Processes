import numpy as np
import time

class QLearning:
    """
    Implements Q-learning algorithm with epsilon-greedy exploration

    If learning_rate is None; alpha(x,a) = 1/max(1, N(s,a))**alpha
    """
    def __init__(self, mdp, gamma=0.99, epsilon=1, epsilon_decay=0.9995,
                 epsilon_min=0, c = 1, reset_randomness = 0.0, seed=42):
        self.mdp = mdp
        self.gamma = gamma
        self.c = c
        self.n_episodes = 0
        self.learning_rate = 1
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((mdp.n_states, mdp.n_actions))
        self.Q[self.mdp.n_nonterminal_states:] = self.mdp.R[self.mdp.n_nonterminal_states:]
        self.Nsa = np.zeros((mdp.n_states, mdp.n_actions))
        self.state = self.mdp.s0
        self.r = 0
        self.reset_randomness = reset_randomness
        self.RS = np.random.RandomState(seed)
        self.episode_end = False

    def get_delta(self, r, x, a, y):
        """
        :param r: reward
        :param x: current state
        :param a: current action
        :param y: next state
        :return:
        """
        max_q_y_a = self.Q[y, :].max()
        q_x_a = self.Q[x, a]

        return r + self.gamma * max_q_y_a - q_x_a

    def get_action(self, state):
        if self.RS.uniform(0, 1) < self.epsilon:
            # Explore
            return np.random.choice(self.mdp.n_actions)
        else:
            # Exploit
            return self.Q[state, :].argmax()
        
    def step(self):

        # Take mdp action
        a  = self.get_action(self.state)
        next_state, self.r, self.episode_end = self.mdp.act(self.state, a)

        # Get Delta Update
        delta = self.get_delta(self.r, self.state, a, next_state)

        # Update Q
        self.Q[self.state, a] = self.Q[self.state, a] + self.learning_rate * delta

        # Keep track of state-action visits
        self.Nsa[self.state, a] += 1

        self.state = next_state

        if self.episode_end:
            self.r += np.min(self.mdp.R[self.state])
            self.n_episodes += 1
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.learning_rate = self.c / (self.c + self.n_episodes)
            self.state = np.random.choice(self.mdp.n_nonterminal_states) if np.random.rand() < self.reset_randomness else self.mdp.s0


def Qlearning_training(qlearning, n_steps=int(5e5), V=None):
    tt = 0
    l0 = 0
    lengths = []
    cumulative_reward = 0
    rewards = np.zeros(n_steps)
    throughputs = np.zeros(n_steps)

    V_error = np.zeros((n_steps))
    start_time = time.time()
    while tt < n_steps:
        qlearning.step()
        # Store estimate of Q*
        if tt < len(V_error): 
            V_est = qlearning.Q.max(axis=1)
            V_error[tt] = np.mean(np.square(V_est - V))
        cumulative_reward += qlearning.r
        tt +=1

        if qlearning.episode_end:
            lengths.append((tt-l0))
            rewards[l0:tt] = cumulative_reward
            throughputs[l0:tt] = -1/(cumulative_reward)
            l0 = tt
            cumulative_reward = 0
            
        if tt % 10000 == 0:

            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / tt) * n_steps
            estimated_remaining_time = estimated_total_time - elapsed_time

            print(f"Step: {tt}/{n_steps}, Time: {elapsed_time/60:.2f}m, ETA: {estimated_remaining_time/60:.2f}m")

    if tt != l0:
        rewards[l0:tt] = cumulative_reward
        throughputs[l0:tt] = -1/(cumulative_reward)

    Q_est = qlearning.Q

    # Compute greedy policy (with estimated Q)
    greedy_policy = np.argmax(qlearning.Q, axis=1)

    return Q_est, V_error, greedy_policy, throughputs, rewards

def print_Qlearning(Q_opt, Q_est, mdp):

    for state in range(mdp.n_states):
        print("state:", mdp.states[state])
        print("true: ", Q_opt[state, :])
        print("est: ", Q_est[state, :])
        print("----------------------------")
