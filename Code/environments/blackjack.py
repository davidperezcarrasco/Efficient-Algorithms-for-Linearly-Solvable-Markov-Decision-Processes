import numpy as np
from frameworks.mdp import MDP
import matplotlib.pyplot as plt
from gym.envs.toy_text import BlackjackEnv


class Black_Jack_MDP(MDP):

    def __init__(self, dynamics = None, gamma=0.95):
        """Initialize the BlackJack MDP."""

        n_states, n_terminal_states = self._create_environment()
        super().__init__(n_states = n_states, n_terminal_states = n_terminal_states, n_actions = len(self.actions), gamma = gamma, s0 = 0)

        if dynamics is None:
            self._create_P()
            self._reward_function()
        else:
            self.P = dynamics['P']
            self.R = dynamics['R']
        

    def _create_environment(self):
        """Create the BlackJack environment."""

        self.env = BlackjackEnv(render_mode="rgb_array", natural=False, sab=True)
        self.env.reset()

        self.deck = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 4}

        nonterminal_states = []
        terminal_states = []
        
        for sum in range(4, 32):
            for ace in range(2):
                for dealer in range(1, 11):
                    if self._is_terminal((sum, dealer, ace, 0, 0)):
                        terminal_states.append((sum, dealer, ace, 0, 0))
                    else:
                        nonterminal_states.append((sum, dealer, ace, 0, 0))
                        nonterminal_states.append((sum, dealer, ace, 0, 1))
                        if dealer == 1:
                            nonterminal_states.append((sum, dealer, ace, 1, 1))
                        else:
                            nonterminal_states.append((sum, dealer, ace, 1, 0))

                if sum < 22:              
                    for dealer in range(11, 23):
                        for dealer_ace in range(2):
                            state = (sum, dealer, ace, 0, dealer_ace)
                            if self._is_terminal(state):
                                terminal_states.append(state)
                            else:
                                nonterminal_states.append(state)
   

        self.states = nonterminal_states + terminal_states
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.actions = list(range(2))
        return len(self.states), len(terminal_states)
    
    def _create_P(self):
        """Create the transition matrix for a deterministic MDP."""

        for sum in range(4, 22):
            for dealer in range(1, 11):
                for ace in range(2):
                    dealer_ace = 1 if dealer == 1 else 0
                    state = (sum, dealer, ace, 1, dealer_ace)
                    index = self.state_to_index[state]
                    terminal_index = self.state_to_index[(sum, dealer, ace, 0, dealer_ace)]
                    self.P[index, 0, terminal_index] = 1
                    for card in self.deck:
                        ac = 1 if card == 1 else ace
                        next_state = (sum + card, dealer, ac, int(sum + card <= 21), dealer_ace if sum + card <= 21 else 0)
                        next_index = self.state_to_index[next_state]
                        self.P[index, 1, next_index] += self.deck[card] / 13

            for dealer in range(1, 17):
                for ace in range(2):
                    for dealer_ace in range(2):
                        dealer_state = (sum, dealer, ace, 0, dealer_ace)
                        dealer_index = self.state_to_index[dealer_state]
                        if dealer >= 7 and dealer <= 11 and dealer_ace == 1:
                            dealer_next_state = (sum, dealer+10, ace, 0, dealer_ace)
                            dealer_next_index = self.state_to_index[dealer_next_state]
                            self.P[dealer_index, 1, dealer_next_index] = 1
                            self.P[dealer_index, 0, dealer_next_index] = 1
                        else:
                            for card in self.deck:
                                d_ace = 1 if card == 1 else dealer_ace
                                next_dealer_state = (sum, min(dealer + card,22), ace, 0, d_ace)
                                next_dealer_index = self.state_to_index[next_dealer_state]
                                self.P[dealer_index, 1, next_dealer_index] += self.deck[card] / 13
                                self.P[dealer_index, 0, next_dealer_index] += self.deck[card] / 13
                    


    def _reward_function(self):
        """Create the reward function for the minigrid environment."""

        for state in range(self.n_nonterminal_states):
            self.R[state] = -1.0

        for state in range(self.n_nonterminal_states, self.n_states):
            sum, dealer, ace, _, dealer_ace = self.states[state]
            if sum > 21:
                self.R[state, 0] = -500.0
                self.R[state, 1] = -500.0
            elif dealer > 21:
                self.R[state, 0] = 0.0
                self.R[state, 1] = 0.0
            else:
                if ace == 1:
                    sum = sum + 10 if sum + 10 <= 21 else sum
                if dealer_ace == 1:
                    dealer = dealer + 10 if dealer + 10 <= 21 else dealer
                if sum > dealer:
                    self.R[state, 0] = 0.0
                    self.R[state, 1] = 0.0
                elif sum < dealer:
                    self.R[state, 0] = -500.0
                    self.R[state, 1] = -500.0
                else:
                    self.R[state, 0] = -1.0
                    self.R[state, 1] = -1.0
            


    def _is_terminal(self, state: tuple[int, int, int, int, int]) -> bool:
        """Check if the state is terminal."""

        return state[0] > 21 or state[1] >= 17
    
    def render(self):
        """Render the environment."""

        image = self.env.render()
        plt.imshow(image)
        plt.show()
    
    def play(self, policy, render=True):
        """Play the game using the given policy."""

        done = False
        obs, _ = self.env.reset()
        a = 0
        while a != -1:

            if done:
                if reward == -1:
                    print("Player loses.")
                elif reward == 0:
                    print("Player draws.")
                else:
                    print("Player wins.")
                obs, _ = self.env.reset()
                done = False
            
            sum, dealer, ace = obs
            dealer_ace = 1 if dealer == 1 else 0
            state = (sum, dealer, int(ace), 1, dealer_ace)
            print("Player's hand:", sum, "Dealer's card:", dealer, "Usable ace:", bool(ace))
            index = self.state_to_index[state]

            if render:
                self.render()

            a = policy[index]

            if a == -1:
                break
            elif a == 0:
                print("Player sticks.")
            else:
                print("Player hits.")

            obs, reward, done, _, _ = self.env.step(a)

                