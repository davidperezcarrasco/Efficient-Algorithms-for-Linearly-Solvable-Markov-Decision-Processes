import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import OrderEnforcing
from environments.grid import CustomEnv
from scipy.sparse import csr_matrix
import frameworks
from frameworks.mdp import MDP
from frameworks.lmdp import LMDP


class Minigrid_MDP(MDP):

    DIR_TO_VEC = [
        # Pointing right (positive X)
        np.array((1, 0)),
        # Down (positive Y)
        np.array((0, 1)),
        # Pointing left (negative X)
        np.array((-1, 0)),
        # Up (negative Y)
        np.array((0, -1)),
    ]

    def __init__(self, grid_size = 14, objects = {}, map = None, dynamics = None, gamma = 0.95):
        """Initialize the Minigrid MDP."""

        self.grid_size = grid_size
        self.n_orientations = 4
        self.actions = list(range(3))
        self.J = {"goal": 0, "lava": -grid_size*6} # Determine reward function for terminal states
        n_states, n_terminal_states = self._create_environment(grid_size, objects, map=map)

        super().__init__(n_states = n_states, n_terminal_states = n_terminal_states, n_actions = len(self.actions), gamma = gamma, s0 = self.s0)
        self.n_cells = int(self.n_states / self.n_orientations)

        if dynamics is None:
            self._create_P()
            self._reward_function()
        else:
            self.P = dynamics['P']
            self.R = dynamics['R']

    def _create_environment(self, grid_size, objects, map=map):
        """Create the Minigrid environment."""

        self.env = OrderEnforcing(CustomEnv(size=grid_size+2, objects=objects, map=map,render_mode="rgb_array"))
        self.env.reset()

        nonterminal_states = []
        terminal_states = []
        for x in range(1, self.env.grid.height):
            for y in range(1, self.env.grid.width):
                if self._is_valid_position(x, y):
                    for o in range(4):
                        state = (x, y, o)
                        if self._is_terminal(state):
                            terminal_states.append(state)
                        else:
                            nonterminal_states.append(state)

        self.states = nonterminal_states + terminal_states
        assert self.grid_size * self.grid_size - len(self.env.walls) == len(self.states) / self.n_orientations, "Invalid number of states"
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.s0 = self.state_to_index[(self.env.agent_start_pos[0], self.env.agent_start_pos[1], self.env.agent_start_dir)]
        return len(self.states), len(terminal_states)

    def _create_P(self):
        """Create the transition matrix for a deterministic MDP."""

        for state in range(self.n_nonterminal_states):
            for action in self.actions:
                next_state = self._state_step(self.states[state], action)
                self.P[state][action][self.state_to_index[next_state]] = 1.0

    def _reward_function(self, uniform_reward=True):
        """Create the reward function for a deterministic MDP.
        
        :uniform_reward: Whether all actions lead to the same reward or it depends on their next state."""

        for state in range(self.n_nonterminal_states):
            for action in self.actions:
                if uniform_reward:
                    self.R[state][action] = -1.0
                else:
                    next_state = self._state_step(self.states[state], action)
                    pos = self.env.grid.get(next_state[0], next_state[1])
                    self.R[state][action] = -1.0 if pos is None or pos.type != "goal" else 0.0
        
        for state in range(self.n_nonterminal_states, self.n_states):
            self.R[state] = self.J[self._state_type(self.states[state])]


    def _is_valid_position(self, x: int, y: int) -> bool:
        """Testing whether a coordinate is a valid location."""

        return (
            0 < x < self.env.width and
            0 < y < self.env.height and
            (self.env.grid.get(x, y) is None or self.env.grid.get(x, y).can_overlap())
        )
    
    def _state_type(self, state: tuple[int, int, int]) -> str:
        """Return the type of a state."""

        pos = self.env.grid.get(state[0], state[1])
        return pos.type if pos is not None else None
    
    def _is_terminal(self, state: tuple[int, int, int]) -> bool:
        """Check if a state is terminal."""

        at_terminal = self._state_type(state) in self.J.keys()

        return at_terminal
    
    def is_goal(self, s: int) -> bool:
        """Check if a state is a goal."""

        state = self.states[s]
        return self._state_type(state) == "goal"

    def _state_step(self, state: tuple[int, int, int], action: int) -> tuple[int, int, int]:
        """Utility to move states one step forward, no side effect."""

        x, y, direction = state

        # Default transition to the sink failure state
        assert self._is_valid_position(x, y)

        # Transition left
        if action == self.env.actions.left:
            direction = (direction - 1) % 4
        # Transition right
        elif action == self.env.actions.right:
            direction = (direction + 1) % 4
        # Transition forward
        elif action == self.env.actions.forward:
            fwd_pos = np.array((x, y)) + self.DIR_TO_VEC[direction]
            if self._is_valid_position(*fwd_pos):
                x, y = fwd_pos
        # Error
        else:
            print(action)
            assert False, "Invalid action"

        return x, y, direction
    
    # Core Methods

    def reset(self, **kwargs):
        """Reset the environment to the initial state. Used mainly for framework rendering."""

        state = self.s0
        self.env.reset(**kwargs)
        if state != self.state_to_index[tuple(np.array((*self.env.unwrapped.agent_start_pos, self.env.unwrapped.agent_start_dir), dtype=np.int32))]:
            self.env.unwrapped.agent_pos = self.states[state][:2]
            self.env.unwrapped.agent_dir = self.states[state][2]
        assert self.state_to_index[tuple(self.observation())] == state
        return state
    
    def step(self, state, action):
        """Step function to interact with the environment."""

        next_state, reward, done = self.act(state, action)
        for a in self.actions:
            if self._state_step(self.states[state], a) == self.states[next_state]:
                self.env.step(a)
        return next_state, reward, done

    def observation(self): 
        """Transform observation."""

        obs = (*self.env.agent_pos, self.env.agent_dir)
        return np.array(obs, dtype=np.int32)
    
    def render(self):
        """Render the environment."""

        image = self.env.render()
        plt.imshow(image)
        plt.show()
    
    def embedding_to_LMDP(self):
        """Embed the Minigrid MDP into a Minigrid LMDP."""

        lmdp, embedding_rmse = super().embedding_to_LMDP()
        dynamics = {'P0': lmdp.P0, 'R': lmdp.R}
        objects = {'walls': self.env.walls, 'lavas': self.env.lavas}
        lmdp_minigrid = Minigrid_LMDP(self.grid_size, map=self.env.map, objects=objects, dynamics = dynamics)
        return lmdp_minigrid, embedding_rmse

    # Auxiliary Methods
    
    def print_attributes(self):
        print("Number of states: ", self.n_states)
        print("Number of actions: ", self.n_actions)
        print("Number of grid cells: ", self.n_cells)
    
    def print_environment(self):
        print("States: ", self.states)
        print("Actions: ", self.actions)


class Minigrid_LMDP(LMDP):

    DIR_TO_VEC = [
        # Pointing right (positive X)
        np.array((1, 0)),
        # Down (positive Y)
        np.array((0, 1)),
        # Pointing left (negative X)
        np.array((-1, 0)),
        # Up (negative Y)
        np.array((0, -1)),
    ]

    def __init__(self, grid_size = 14, objects = {}, map=None, dynamics = None, lmbda = 1):
        """Initialize the minigrid environment."""

        self.grid_size = grid_size
        self.n_orientations = 4
        self.J = {"goal": 0, "lava": -grid_size*6} # Determine reward function for terminal states
        n_states, n_terminal_states = self._create_environment(grid_size, objects, map=map)
        
        super().__init__(n_states = n_states, n_terminal_states = n_terminal_states, lmbda=lmbda, s0=self.s0)
        self.n_cells = int(self.n_states / self.n_orientations)
        
        if dynamics is None:
            self._create_P0()
            self._reward_function()
        else:
            self.P0 = dynamics['P0']
            self.R = dynamics['R']

    def _create_environment(self, grid_size, objects, map=map):
        """Create the environment for the minigrid."""

        self.env = OrderEnforcing(CustomEnv(size=grid_size+2, objects=objects, map=map, render_mode="rgb_array"))
        self.env.reset()
        
        nonterminal_states = []
        terminal_states = []
        for x in range(1, self.env.grid.height):
            for y in range(1, self.env.grid.width):
                if self._is_valid_position(x, y):
                    for o in range(4):
                        state = (x, y, o)
                        if self._is_terminal(state):
                            terminal_states.append(state)
                        else:
                            nonterminal_states.append(state)

        self.states = nonterminal_states + terminal_states
        assert self.grid_size * self.grid_size - len(self.env.walls) == len(self.states) / self.n_orientations, "Invalid number of states"
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.s0 = self.state_to_index[(self.env.agent_start_pos[0], self.env.agent_start_pos[1], self.env.agent_start_dir)]
        return len(self.states), len(terminal_states)

    def _create_P0(self, sparse = True):
        """Create the uncontrolled transition probabilities matrix for a stochastic LMDP."""

        transitions = [self.env.actions.left, self.env.actions.right, self.env.actions.forward]
        for state in range(self.n_nonterminal_states):
            for t in transitions:
                next_state = self.state_step(self.states[state], t)
                self.P0[state][self.state_to_index[next_state]] += 1/len(transitions)
        if sparse:
            self.P0 = csr_matrix(self.P0)

    def _reward_function(self):
        """Create the reward function for the minigrid environment."""

        for state in range(self.n_nonterminal_states):
            self.R[state] = -1.0

        for state in range(self.n_nonterminal_states, self.n_states):
            self.R[state] = self.J[self._state_type(self.states[state])]

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Testing whether a coordinate is a valid location."""

        return (
            0 < x < self.env.width and
            0 < y < self.env.height and
            (self.env.grid.get(x, y) is None or self.env.grid.get(x, y).can_overlap())
        )
    
    def _state_type(self, state: tuple[int, int, int]) -> str:
        """Return the type of a state."""

        pos = self.env.grid.get(state[0], state[1])
        return pos.type if pos is not None else None
    
    def _is_terminal(self, state: tuple[int, int, int]) -> bool:
        """Check if a state is terminal."""

        at_terminal = self._state_type(state) in self.J.keys()
        return at_terminal
    
    def is_goal(self, s: int) -> bool:
        """Check if a state is a goal."""

        state = self.states[s]
        return self._state_type(state) == "goal"

    def state_step(self, state: tuple[int, int, int], action: int) -> tuple[int, int, int]:
        """Utility to move states one step forward, no side effect."""
        
        x, y, direction = state

        # Default transition to the sink failure state
        assert self._is_valid_position(x, y)

        # Transition left
        if action == self.env.actions.left:
            direction = (direction - 1) % 4
        # Transition right
        elif action == self.env.actions.right:
            direction = (direction + 1) % 4
        # Transition forward
        elif action == self.env.actions.forward:
            fwd_pos = np.array((x, y)) + self.DIR_TO_VEC[direction]
            if self._is_valid_position(*fwd_pos):
                x, y = fwd_pos
        # Error
        else:
            print(action)
            assert False, "Invalid action"

        return x, y, direction
    
    # Core Methods

    def reset(self, **kwargs):
        """Reset the environment to the initial state. Used mainly for framework rendering."""

        state = self.s0
        self.env.reset(**kwargs)
        if state != self.state_to_index[tuple(np.array((*self.env.unwrapped.agent_start_pos, self.env.unwrapped.agent_start_dir), dtype=np.int32))]:
            self.env.unwrapped.agent_pos = self.states[state][:2]
            self.env.unwrapped.agent_dir = self.states[state][2]
        assert self.state_to_index[tuple(self.observation())] == state
        return state
    
    def step(self, state, P):
        """Step function to interact with the environment."""

        next_state, reward, done = self.act(state, P)
        for action in range(3):
            if self.state_step(self.states[state], action) == self.states[next_state]:
                self.env.step(action)
        return next_state, reward, done

    def observation(self):
        """Transform observation."""

        obs = (*self.env.agent_pos, self.env.agent_dir)
        return np.array(obs, dtype=np.int32)
    
    def render(self):
        """Render the environment."""

        image = self.env.render()
        plt.imshow(image)
        plt.show()
    
    def embedding_to_MDP(self):
        """Embed the Minigrid lMDP into a Minigrid MDP."""

        mdp, embedding_mse = super().embedding_to_MDP()
        dynamics = {'P': mdp.P, 'R': mdp.R}
        objects = {'walls': self.env.walls, 'lavas': self.env.lavas}
        mdp_minigrid = Minigrid_MDP(self.grid_size, map=self.env.map, objects=objects, dynamics = dynamics)
        return mdp_minigrid, embedding_mse
    
    # Auxiliary Methods
    
    def print_attributes(self):
        print("Number of states: ", self.n_states)
        print("Number of actions: ", self.n_actions)
        print("Number of grid cells: ", self.n_cells)
    
    def print_environment(self):
        print("States: ", self.states)
        print("Actions: ", self.actions)
