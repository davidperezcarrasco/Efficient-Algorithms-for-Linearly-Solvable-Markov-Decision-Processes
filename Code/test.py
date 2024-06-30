from environments.minigrids import Minigrid_LMDP, Minigrid_MDP
from environments.blackjack import Black_Jack_MDP
import numpy as np
import time
from utils.plot import Plotter, Minigrid_MDP_Plotter
from algs.zlearning import ZLearning, Zlearning_training
from algs.qlearning import QLearning, Qlearning_training
from utils.lmdp_plot import Minigrid_LMDP_Plotter
from scipy.sparse import csr_matrix
import cProfile


import gymnasium as gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    grid_size = 15
    walls = []#(14,1), (1,8), (5, 5), (12, 5), (8, 7), (2,5), (3,5), (4,5), (6,5), (7,5), (8,5), (9,5), (10,5), (11,5), (13,5), (15,9)]
    lavas = []#(2,1)]#(7,1), (1,4), (1,14)]
    objects = {"walls": walls, "lavas": lavas}

    simple15_map = [
        "########",
        "#A     #",
        "#      #",
        "#  W   #",
        "#      #",
        "#      #",
        "#L    G#",
        "########"
    ]

    hill15_map = [
        "#################",
        "#A              #",
        "# WLLLLLLLLLLLL #",
        "# WWWWWWWWWWWWL #",
        "# W   W   W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "# W W W W W  WL #",
        "#   W   W      G#",
        "#################"
    ]

    maze13_map = [
        "###############",
        "#A    W       #",
        "# W W W W W W #",
        "# W   W W W W #",
        "# W W W W W W #",
        "# W W W W   W #",
        "# W W W WWWWW #",
        "# W     W     #",
        "# W WWWWW WWW #",
        "# W W   W   W #",
        "# WWW W W W W #",
        "#     W W W W #",
        "# WWWWW W W WW#",
        "#     W W    G#",
        "###############"
    ]

    maze20_map = [
        "####################",
        "#A    W     W     G#",
        "#W W  W W W W W W  #",
        "#W W      W W W W  #",
        "#W W W W WWW  WWWWW#",
        "#W   W W   W   W   #",
        "#WWWWWWWWW WWW WW W#",
        "#  W   W     W W   #",
        "#  W W WWW W W W  W#",
        "#  W W   W W W     #",
        "#  W WW    WWWWW WW#",
        "#    W       W W W #",
        "#WWWWW WWWWW W W W #",
        "#W     W     W   W #",
        "#W WWWWW WWWWW WWW #",
        "#W     W     W     #",
        "#WWWWWWW WWWWWWW WW#",
        "#        W        W#",
        "#W W W W   W W W   #",
        "####################"
    ]

    rooms_map = [
        "######################",
        "#             W     G#",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#WWWW WW WWWWWWW WWWWW#",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W  A   W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#W WWWWWWWWWWWWWWWWW W#",
        "#      W             #",
        "#      W      W      #",
        "#      W      W      #",
        "#      W      W      #",
        "#             W      #",
        "#      W      W      #",
        "######################"
    ]

    grid_map = None
    grid_size = len(grid_map)-2 if grid_map is not None else grid_size

    gamma = 1
    epsilon = 1e-10
    n_iters = int(4e4)
    lmbda = 1

    # MDP
    minigrid_mdp = Minigrid_MDP(grid_size=grid_size, objects=objects, map=grid_map, gamma=gamma)
    minigrid_mdp_plots = Minigrid_MDP_Plotter(minigrid_mdp)
    minigrid_mdp.render()
    minigrid_lmdp = Minigrid_LMDP(grid_size=grid_size, objects=objects, map=grid_map, lmbda=lmbda)
    minigrid_lmdp, error = minigrid_mdp.embedding_to_LMDP()
    print(error)
    Z, n_steps = minigrid_lmdp.power_iteration(lmbda = lmbda, epsilon=epsilon)
    