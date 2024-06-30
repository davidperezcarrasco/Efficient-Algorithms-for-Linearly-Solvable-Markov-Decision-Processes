import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from utils.plot import Plotter

class Minigrid_LMDP_Plotter(Plotter):
    def __init__(self, lmdp):
        super().__init__()
        self.lmdp = lmdp
    
    def show_Z(self, Z, print_Z=False, plot_Z=True, file = "Z_value.txt"):
        if print_Z:
            with open(file, "w") as f:
                for s in range(self.lmdp.n_states):
                    #print("Z value in state ", lmdp.states[s], ": ", Z[s])
                    f.write("Z({}): {}\n".format(self.lmdp.states[s], Z[s]))
        if plot_Z:
            plt.figure(figsize=(5, 2*self.lmdp.grid_size))
            im = plt.imshow(Z.reshape((self.lmdp.n_cells,4)), cmap='viridis', origin='lower')
            plt.title("Z(i)")
            plt.xlabel("Orientation")
            plt.ylabel("Cell Position")
            y_labels = [f"{i//self.lmdp.grid_size + 1}x{i%self.lmdp.grid_size + 1}" for i in range(self.lmdp.n_cells)]
            # Adjust the range of values for the colorbar legend
            colorbar_min = Z.min() 
            colorbar_max = Z.max()
            colorbar_ticks = np.linspace(colorbar_min, colorbar_max, 10)
            x_labels = [f"{i}" for i in range(self.lmdp.n_orientations)]
            plt.yticks(range(self.lmdp.n_cells), y_labels)
            plt.xticks(range(self.lmdp.n_orientations), x_labels)
            plt.colorbar(im, ticks=colorbar_ticks)  # Add colorbar for reference
            plt.show()

    def show_Pu(self, PU, print_Pu=False, plot_Pu=True, is_sparse=False):
        if print_Pu:
            for i in range(self.lmdp.n_states):
                for j in (PU[i].indices if is_sparse else range(self.lmdp.n_states)):
                    if PU[i,j] != 0: print("Pu[", self.lmdp.states[i], "][", self.lmdp.states[j], "]: ", PU[i,j])

        if plot_Pu:
            if is_sparse: PU = PU.todense()
            # Create a heatmap
            plt.figure(figsize=(15*self.lmdp.grid_size, 10*self.lmdp.grid_size))
            plt.imshow(PU, cmap='viridis', origin='lower')

            # Add colorbar for reference
            cbar = plt.colorbar()
            cbar.set_label("Pij(u*)")

            # Add title and labels
            plt.title("Controlled Transition Probabilities")
            plt.xlabel("Output State j")
            plt.ylabel("Input State i")
            plt.yticks(range(self.lmdp.n_states), self.lmdp.states, fontsize=10)
            plt.xticks(range(self.lmdp.n_states), self.lmdp.states, fontsize=10, rotation='vertical')

            plt.show()

    def plot_sample_path(self, PU, start = 0, path = 'LMDP_PU_path.gif'):
        self.lmdp.s0 = start if self.lmdp.s0 != start else self.lmdp.s0
        s = self.lmdp.reset()
        done = False
        with imageio.get_writer(path, mode='I', duration=0.2, loop=10) as writer:
            writer.append_data(self.lmdp.env.render())
            while not (done and self.lmdp.is_goal(s)):
                s, _, done = self.lmdp.step(s, PU) if not self.lmdp.is_goal(s) else (start, 0, False)
                writer.append_data(self.lmdp.env.render())
        os.startfile(path) #for windows 

    def compare_Zlearning_estimates(self, zlearning, Z_opt, Pu_opt):
        for i in range(self.lmdp.n_states):
            print("state:", self.lmdp.states[i])
            print("true: ", Z_opt[i])
            print("est: ", zlearning.Z[i])
            print("----------------------------")

        self.show_Pu(self, zlearning.Pu, print_Pu = True, plot_Pu = False, is_sparse = True)
        print("Total Absolute Error: ", np.sum(np.abs(Pu_opt[0:-4]-zlearning.Pu[0:-4])))