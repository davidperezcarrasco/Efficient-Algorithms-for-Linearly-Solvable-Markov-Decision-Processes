import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import pandas as pd
import math
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors

class Plotter:
    def __init__(self):
        try:
            # Try to enable TeX rendering
            plt.rcParams["text.usetex"] = True
        except FileNotFoundError as e:
            # If a FileNotFoundError occurs (related to TeX), disable TeX rendering
            print("Warning: TeX rendering unavailable. Using Matplotlib's default fonts.")
            plt.rcParams["text.usetex"] = False
        
        sns.set_context(context="paper", font_scale=1.2)
        sns.set_style({'font.family': 'serif'})

    def plot_throughput(self, throughputs, grid_size, names, smooth_window = 10000, save_path = 'plots\'', title="Episodic Throughput in Minigrid "):
        df = pd.DataFrame()
        for i in range(len(names)):
            temp_df = pd.DataFrame()
            temp_df['rewards'] = throughputs[i]
            temp_df['index'] = range(len(throughputs[i]))
            temp_df['name'] = names[i]
            df = pd.concat([df, temp_df.reset_index(drop=True)], ignore_index=True)

        # Calculate rolling mean and standard deviation
        df['smoothed_rewards'] = df.groupby('name')['rewards'].transform(lambda x: x.rolling(window=smooth_window, min_periods=1, center=True).mean())
        df['smoothed_std'] = df.groupby('name')['rewards'].transform(lambda x: x.rolling(window=smooth_window, min_periods=1, center=True).std())

        plt.figure(figsize=(8, 6))

        for name in names:
            subset = df[df['name'] == name]
            plt.plot(subset['index'], subset['smoothed_rewards'], label=name)
            max_val = subset['smoothed_rewards'].max()
            min_val = subset['smoothed_rewards'].min()
            
            # Clip the upper and lower bounds of the confidence interval to the max and min values
            upper_bound = np.clip(subset['smoothed_rewards'] + subset['smoothed_std'], min_val, max_val)
            lower_bound = np.clip(subset['smoothed_rewards'] - subset['smoothed_std'], min_val, max_val)
            
            plt.fill_between(subset['index'], lower_bound, upper_bound, alpha=0.2)
        

            
        ax = plt.gca()
        ax.set(xlabel="Time Step", ylabel="Episodic Throughput")
        legend = plt.legend(loc='upper right', fontsize=12)
        legend.get_frame().set_edgecolor('black') 
        plt.title(title)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(save_path + str(grid_size) + 'throughputs.png')
        plt.show()
        plt.clf()

    def plot_rewards(self, rewards, grid_size, names, smooth_window_initial=1, smooth_window_later=2000, save_path='plots/', title="Episodic Reward in Minigrid "):
        df = pd.DataFrame()
        for i in range(len(names)):
            temp_df = pd.DataFrame()
            temp_df['rewards'] = rewards[i]
            temp_df['index'] = range(len(rewards[i]))
            temp_df['name'] = names[i]
            df = pd.concat([df, temp_df.reset_index(drop=True)], ignore_index=True)

        def combined_smoothing(x, transition_point):
            smoothed = np.empty_like(x)
            smoothed[:transition_point] = x[:transition_point].rolling(window=smooth_window_initial, min_periods=1, center=False).mean()
            smoothed[transition_point:] = x[transition_point:].rolling(window=smooth_window_later, min_periods=1, center=True).mean()
            return smoothed

        # Calculate rolling mean and standard deviation using combined smoothing
        df['smoothed_rewards'] = df.groupby('name')['rewards'].transform(lambda x: combined_smoothing(x, smooth_window_later-500))
        df['smoothed_std'] = df.groupby('name')['rewards'].transform(lambda x: x.rolling(window=smooth_window_later, min_periods=1, center=True).std())

        plt.figure(figsize=(8, 6))

        for name in names:
            subset = df[df['name'] == name]
            plt.plot(subset['index'], subset['smoothed_rewards'], label=name)
            max_val = subset['smoothed_rewards'].max()
            min_val = subset['smoothed_rewards'].min()

            # Clip the upper and lower bounds of the confidence interval to the max and min values
            upper_bound = np.clip(subset['smoothed_rewards'] + subset['smoothed_std'], min_val, max_val)
            lower_bound = np.clip(subset['smoothed_rewards'] - subset['smoothed_std'], min_val, max_val)

            plt.fill_between(subset['index'], lower_bound, upper_bound, alpha=0.2)

        ax = plt.gca()
        ax.set(xlabel="Time Step", ylabel="Episodic Reward (log scale)")
        plt.yscale('symlog')
        legend = plt.legend(loc='upper right', fontsize=12)
        legend.get_frame().set_edgecolor('black')
        plt.title(title)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(save_path + str(grid_size) + 'rewards.png')
        plt.show()
        plt.clf()

    def plot_rewards_and_errors(self, errors, rewards, names, smooth_window_initial=2000, smooth_window_later=2000, save_path='plots/', title="Episodic Reward in Minigrid domain"):
        # Prepare data for errors
        df_errors = pd.DataFrame()
        # Prepare data for rewards
        df_rewards = pd.DataFrame()
        for i in range(len(names)):
            temp_df = pd.DataFrame()
            temp_df['errors'] = errors[i]
            temp_df['index'] = range(len(errors[i]))
            temp_df['name'] = names[i]
            df_errors = pd.concat([df_errors, temp_df.reset_index(drop=True)], ignore_index=True)

            temp_df2 = pd.DataFrame()
            temp_df2['rewards'] = rewards[i]
            temp_df2['index'] = range(len(rewards[i]))
            temp_df2['name'] = names[i]
            df_rewards = pd.concat([df_rewards, temp_df2.reset_index(drop=True)], ignore_index=True)

        def combined_smoothing(x, transition_point):
            smoothed = np.empty_like(x)
            smoothed[:transition_point] = x[:transition_point].rolling(window=smooth_window_initial, min_periods=1, center=False).mean()
            smoothed[transition_point:] = x[transition_point:].rolling(window=smooth_window_later, min_periods=1, center=True).mean()
            return smoothed

        # Calculate rolling mean and standard deviation using combined smoothing
        df_rewards['smoothed_rewards'] = df_rewards.groupby('name')['rewards'].transform(lambda x: combined_smoothing(x, smooth_window_later-500))

        # Set up the subplots
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))

        # Plotting errors
        sns.lineplot(x='index', y='errors', data=df_errors, hue='name', style='name', ax=axs[0])
        axs[0].set(ylabel="Approximation Error (log scale)")
        axs[0].set_yscale('log')
        handles, labels = axs[0].get_legend_handles_labels()
        new_labels = [label for label in labels]
        axs[0].legend(handles, new_labels, loc='upper right', fontsize=12).get_frame().set_edgecolor('black')
        axs[0].set_title(title)
        axs[0].grid(True, which="both", ls="--", linewidth=0.5)
        axs[0].set_xlabel("")

        sns.lineplot(x='index', y='smoothed_rewards', data=df_rewards, hue='name', style='name', ax=axs[1])
        axs[1].set(xlabel="Time Step", ylabel="Episodic Reward (symlog scale)")
        axs[1].set_yscale('symlog')
        axs[1].set_ylim([min(df_rewards['smoothed_rewards'].min()-500, 1e-3), max(df_rewards['smoothed_rewards'].max()+5, -10)])  # Adjust y-axis limits here
        handles, labels = axs[1].get_legend_handles_labels()
        new_labels = [label for label in labels]
        axs[1].legend(handles, new_labels, loc='upper right', fontsize=12).get_frame().set_edgecolor('black')
        #axs[1].legend(loc='upper right', fontsize=12).get_frame().set_edgecolor('black')
        axs[1].grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path + 'rewards_and_errors.png')
        plt.show()
        plt.clf()

    def plot_embedding_error_scatter(self, mdp, embedded_lmdps, embedding_names, gamma = 1, lmbda = 1, save_path = 'plots\''):
        mdp_V = mdp.value_iteration(gamma=gamma)[0].max(axis=1)[:mdp.n_nonterminal_states]

        num_lmdps = len(embedded_lmdps)
        cols = 2 if num_lmdps > 3 else min(num_lmdps, 3)  # 2 columns if more than 3, else max 3 columns
        rows = math.ceil(num_lmdps / cols)

        colors = sns.color_palette("tab10", num_lmdps * 2)[:num_lmdps]

        fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), sharey=False)
        axs = axs.flatten() if num_lmdps > 1 else [axs]

        for i, (embedded_lmdp, color, name) in enumerate(zip(embedded_lmdps, colors, embedding_names)):
            lmdp_V = embedded_lmdp.Z_to_V(embedded_lmdp.power_iteration(lmbda=lmbda)[0])[:embedded_lmdp.n_nonterminal_states]
            r_squared = r2_score(mdp_V, lmdp_V)
            sns.scatterplot(x=mdp_V, y=lmdp_V, ax=axs[i], label=name, alpha=0.7, s=20, color=color, edgecolor='black')
            axs[i].set_title(chr(65 + i), fontsize=20, loc='left') 
            if (i // cols) == (rows - 1):  # Only set x labels for the bottom row
                axs[i].set_xlabel('Value in traditional MDP', fontsize=12)
            if i % cols == 0:
                axs[i].set_ylabel('Value in LMDP', fontsize=12)
            axs[i].grid(False)
            axs[i].text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=axs[i].transAxes, fontsize=12, verticalalignment='top')

            # Add diagonal line
            min_val = min(min(mdp_V), min(lmdp_V))
            max_val = max(max(mdp_V), max(lmdp_V))
            axs[i].plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=0.5)

        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        handles, labels = [], []
        for ax in axs:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            ax.legend().set_visible(False)
        fig.legend(handles, labels, loc='upper right', ncol=cols, fontsize=12).get_frame().set_edgecolor('black')

        plt.suptitle('Comparison of Embedding Approaches', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.subplots_adjust(hspace=0.3)
        
        plt.savefig(save_path + 'embedding_error.png')
        plt.show()
        plt.clf()

    def plot_mse_vs_grid_size(self, n_states, mse_values, names, save_path = 'plots/'):
        df = pd.DataFrame()
        for mse, name in zip(mse_values, names):
            temp_df = pd.DataFrame({'Number of States': n_states, 'MSE': mse, 'Method': name, 'Name': name})
            df = pd.concat([df, temp_df], ignore_index=True)
        
        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(data=df, x='Number of States', y='MSE', hue='Name', style='Method', markers=True, dashes=False)

        plt.xlabel('Number of States', fontsize=14)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
        #plt.yscale('log')  # Use logarithmic scale for y-axis
        plt.yscale('log')  # Use logarithmic scale for y-axis
        plt.title('MSE vs. Number of States for Embedding Approaches', fontsize=16)
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        # Customize legend
        legend = plt.legend(loc='upper right', fontsize=12)
        legend.get_frame().set_edgecolor('black')  
        
        
        plt.tight_layout()
        plt.savefig(save_path + 'mse_vs_n_states2.png')
        plt.show()
        plt.clf()

    def plot_values_vs_indices(self, indices, values, names, save_path = 'plots/', scale = None, title="Value Function vs. Number of States"):
        df = pd.DataFrame()
        for value, name in zip(values, names):
            temp_df = pd.DataFrame({'indices': indices, 'values': value, 'Method': name})
            df = pd.concat([df, temp_df], ignore_index=True)
        
        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(data=df, x='indices', y='values', hue='Method', style='Method', markers=False, dashes=False)

        plt.xlabel('Time step', fontsize=14)
        plt.ylabel('Mean Squared Error (log scale)', fontsize=14)
        # if scale is not None:
        #     plt.yscale(scale)
        plt.title(title, fontsize=16)
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        # Customize legend
        legend = plt.legend(loc='upper right', fontsize=12)
        legend.get_frame().set_edgecolor('black')  

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.clf()

    def plot_episode_length(self, lengths, opt_length, plot_batch=False, batch_size=50):
        print("Number of episodes: ", len(lengths))
        plt.plot(range(1, len(lengths)+1), lengths)
        plt.axhline(y=opt_length, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length vs. Episode")
        plt.show()
        print("Last episode length: ",lengths[-1])
        if plot_batch:
            averaged_lengths = [sum(lengths[i:i+batch_size]) / batch_size for i in range(0, len(lengths)-len(lengths)%batch_size, batch_size)]
            averaged_lengths.append(np.mean(lengths[-(len(lengths) % batch_size):]))
            plt.plot(range(1, len(averaged_lengths)+1), averaged_lengths)
            plt.axhline(y=opt_length, color='r', linestyle='--', alpha=0.5)
            plt.xlabel("Batch of Episodes")
            plt.ylabel("Average Episode Length")
            plt.title("Episodes Batch Length vs. Batch of Episodes")
            plt.show()
            print("Last batch averaged length: ", averaged_lengths[-1])

    def plot_episode_throughput(self, throughputs, opt_length = None, smooth_window=5000, title="Throughput vs. Time step"):
        if opt_length is not None: 
            plt.axhline(y=-opt_length, color='r', linestyle='--', alpha=0.5)
        throughputs_series = pd.Series(throughputs)
        smoothed_throughputs = throughputs_series.rolling(window=smooth_window, center=True).mean()
        throughputs = throughputs_series.rolling(window=int(smooth_window/10), center=True).mean()
        plt.plot(range(1, len(throughputs)+1), throughputs, color='b', alpha=0.09)
        plt.plot(range(1, len(smoothed_throughputs)+1), smoothed_throughputs, color='b')
        plt.xlabel("Time step")
        plt.ylabel("Throughput")
        plt.title(title)
        plt.show()

    def plot_episode_reward(self, rewards, opt_length = None, smooth_window=5000, title="Episodic Reward vs. Time step"):
        if opt_length is not None: 
            plt.axhline(y=-opt_length, color='r', linestyle='--', alpha=0.5)
        rewards_series = pd.Series(rewards)
        smoothed_rewards = rewards_series.rolling(window=smooth_window, min_periods=1, center=True).mean()
        rewards = rewards_series.rolling(window=int(smooth_window/10), min_periods=1, center=True).mean()
        plt.yscale('symlog')
        plt.plot(range(1, len(rewards)+1), rewards, color='b', alpha=0.09)
        plt.plot(range(1, len(smoothed_rewards)+1), smoothed_rewards, color='b')
        plt.xlabel("Time step")
        plt.ylabel("Episodic Reward")
        plt.title(title)
        plt.show()
    
            
    def compare_throughputs(self, throughputs, grid_size, names, save_path = 'plots\''):
        for i, throughput in enumerate(throughputs):
            plt.plot(range(1, len(throughput)+1), throughput, label= names[i])
        plt.axhline(y=1/(2*grid_size-1), color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        plt.title("Average Reward per Time Step in Minigrid " + str(grid_size) + "x"+ str(grid_size))
        plt.legend()
        plt.savefig(save_path + str(grid_size) + 'throughputs.png')
        plt.show()

    def plot_values_per_hyperparameter(self, values_list, indices, names, title, xlabel, ylabel, save_path='plots/'):
        for i, values in enumerate(values_list):
            sns.lineplot(x=indices, y=values, label=names[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.legend(loc='upper right', fontsize=12).get_frame().set_edgecolor('black')
        plt.tight_layout()
        plt.savefig(save_path + title + '.png')
        plt.show()
        plt.clf()

    def plot_convergence(self, Opt, Est, model = 'Z-learning'):
        diff = np.abs(Est - Opt).mean(axis=(1))
        plt.plot(diff)
        plt.xlabel('iteration')
        plt.ylabel('Error')
        plt.title(model + " convergence")
        plt.show()

    def plot_grid(self, env, grid_size, value_function, walls, lavas):
        # Initialize the grid for the value function
        grid = np.full((grid_size, grid_size), -np.inf)
        
        for state, value in enumerate(value_function):
            y, x = env.states[state] #Adapt
            grid[x, y] = value
        
        # Initialize the plot
        fig, ax = plt.subplots()

        # Mask the walls for the color mapping
        masked_grid = np.ma.masked_where(grid == -np.inf, grid)

        # Create a colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("", [
            (0.25, 0.25, 0.25),   # Dim grey
            (0.5, 0.5, 0.5),  # Dark grey
            (0.7, 0.7, 0.7),  # Grey
            (0.85, 0.85, 0.85),  # Light grey
            (1, 1, 1)  # White
        ])
        norm = plt.Normalize(vmin=np.min(masked_grid), vmax=np.max(masked_grid))

        # Plot the grid
        for (i, j), value in np.ndenumerate(grid):
            if (i, j) in walls:
                color = 'black'
            elif (i, j) in lavas:
                color = 'lightcoral'
            else:
                color = cmap(norm(value))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))

        # Set the limits and grid
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size + 1, 1))
        ax.set_yticks(np.arange(0, grid_size + 1, 1))
        ax.grid(which='both', color='black')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)

        # Invert Y-axis to match typical matrix layout
        plt.gca().invert_yaxis()

        # Add color bar (legend) for non-wall cells
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(masked_grid)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Value Function')

        plt.show()

    def plot_minigrid(self, env, grid_size, value_function):
        # Initialize the grid for the value function
        grid = np.full((grid_size, grid_size), 0)
        
        for x in range(0, grid_size):
            for y in range(0, grid_size):
                state1 = (x+1,y+1,0)
                state2 = (x+1,y+1,1)
                state3 = (x+1,y+1,2)
                state4 = (x+1,y+1,3)
                if state1 in env.states and env.state_to_index[state1] < env.n_nonterminal_states:
                    grid[x,y] = (value_function[env.state_to_index[state1]]+ value_function[env.state_to_index[state2]]+ value_function[env.state_to_index[state3]]+ value_function[env.state_to_index[state4]])/4
                elif state1 not in env.states:
                    grid[x,y] = 50
                elif not env.is_goal(env.state_to_index[state1]):
                    grid[x,y] = 100
        
        # Initialize the plot
        fig, ax = plt.subplots()

        # Mask the walls for the color mapping
        masked_grid = np.ma.masked_where(grid == -np.inf, grid)

        # Create a colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("", [
            (1, 1, 1),  # White
            (0.85, 0.85, 0.85),  # Light grey
            (0.7, 0.7, 0.7),  # Grey
            (0.5, 0.5, 0.5),  # Dark grey
            (0.25, 0.25, 0.25)   # Dim grey
        ])
        norm = plt.Normalize(vmin=np.min(masked_grid), vmax=0)

        # Plot the grid
        for (i, j), value in np.ndenumerate(grid):
            if value==50:
                color = 'black'
            elif value==100:
                color = 'orange'
            else:
                color = cmap(norm(value))
            ax.add_patch(plt.Rectangle((i, j), 1, 1, color=color))

        # Set the limits and grid
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size + 1, 1))
        ax.set_yticks(np.arange(0, grid_size + 1, 1))
        ax.grid(which='both', color='black')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)

        # Invert Y-axis to match typical matrix layout
        plt.gca().invert_yaxis()

        # Add color bar (legend) for non-wall cells
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(masked_grid)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Value Function')

        plt.show()

    def plot_grids(self, env, grid_size, value_functions, names):
        num_plots = len(value_functions)
        cols = 3
        rows = (num_plots + cols - 1) // cols # Calculate number of rows needed
        fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), sharey=True, sharex=True)
        axs = axs.flatten() if num_plots > 1 else [axs]

        for idx, (value_function, name) in enumerate(zip(value_functions, names)):
            # Initialize the grid for the value function
            grid = np.full((grid_size, grid_size), 0)

            for x in range(0, grid_size):
                for y in range(0, grid_size):
                    state1 = (x+1, y+1, 0)
                    state2 = (x+1, y+1, 1)
                    state3 = (x+1, y+1, 2)
                    state4 = (x+1, y+1, 3)
                    if state1 in env.states and env.state_to_index[state1] < env.n_nonterminal_states:
                        grid[x, y] = (value_function[env.state_to_index[state1]] + 
                                    value_function[env.state_to_index[state2]] + 
                                    value_function[env.state_to_index[state3]] + 
                                    value_function[env.state_to_index[state4]]) / 4
                    elif state1 not in env.states:
                        grid[x, y] = 50
                    elif not env.is_goal(env.state_to_index[state1]):
                        grid[x, y] = 100

            # Mask the walls for the color mapping
            masked_grid = np.ma.masked_where(grid == -np.inf, grid)

            # Create a colormap
            cmap = mcolors.LinearSegmentedColormap.from_list("", [
                (1, 1, 1),  # White
                (0.85, 0.85, 0.85),  # Light grey
                (0.7, 0.7, 0.7),  # Grey
                (0.5, 0.5, 0.5),  # Dark grey
                (0.25, 0.25, 0.25)  # Dim grey
            ])
            norm = plt.Normalize(vmin=np.min(masked_grid), vmax=0)

            # Plot the grid
            ax = axs[idx]
            for (i, j), value in np.ndenumerate(grid):
                if value == 50:
                    color = 'black'
                elif value == 100:
                    color = 'orange'
                else:
                    color = cmap(norm(value))
                ax.add_patch(plt.Rectangle((i, j), 1, 1, color=color))

            # Set the limits and grid
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_xticks(np.arange(0, grid_size + 1, 1))
            ax.set_yticks(np.arange(0, grid_size + 1, 1))
            ax.grid(which='both', color='black')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)

            # Invert Y-axis to match typical matrix layout
            ax.invert_yaxis()

            # Add title
            ax.set_title(chr(65 + idx), fontsize=20, loc='left')

        # Remove empty subplots
        for j in range(idx + 1, len(axs)):
            fig.delaxes(axs[j])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(masked_grid)
        cbar = fig.colorbar(sm, ax=axs[-1], location='right', fraction=0.046, pad=0.04)
        cbar.set_label('Value Function')

        plt.tight_layout()
        plt.show()

    def plot_grids_with_policies(self, env, grid_size, value_functions, names=None):
        num_plots = len(value_functions)
        cols = 3
        rows = (num_plots + cols - 1) // cols  # Calculate number of rows needed
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharey=True, sharex=True)
        axs = axs.flatten() if num_plots > 1 else [axs]

        for idx, (value_function, name) in enumerate(zip(value_functions, names if names is not None else [None] * num_plots)):
            # Initialize the grid for the value function
            grid = np.full((grid_size, grid_size), 0.0)

            for x in range(0, grid_size):
                for y in range(0, grid_size):
                    state1 = (x + 1, y + 1, 0)
                    state2 = (x + 1, y + 1, 1)
                    state3 = (x + 1, y + 1, 2)
                    state4 = (x + 1, y + 1, 3)
                    if state1 in env.states and env.state_to_index[state1] < env.n_nonterminal_states:
                        grid[x, y] = float(value_function[env.state_to_index[state1]] +
                                    value_function[env.state_to_index[state2]] +
                                    value_function[env.state_to_index[state3]] +
                                    value_function[env.state_to_index[state4]]) / 4.0
                    elif state1 not in env.states:
                        grid[x, y] = 50
                    elif not env.is_goal(env.state_to_index[state1]):
                        grid[x, y] = 100

            # Mask the walls for the color mapping
            masked_grid = np.ma.masked_where(grid == -np.inf, grid)

            # Create a colormap
            cmap = mcolors.LinearSegmentedColormap.from_list("", [
                (1, 1, 1),  # White
                (0.85, 0.85, 0.85),  # Light grey
                (0.7, 0.7, 0.7),  # Grey
                (0.5, 0.5, 0.5),  # Dark grey
                (0.25, 0.25, 0.25)  # Dim grey
            ])
            norm = plt.Normalize(vmin=np.min(masked_grid), vmax=0)

            # Plot the grid
            ax = axs[idx]
            for (i, j), value in np.ndenumerate(grid):
                if value == 50:
                    color = 'black'
                elif value == 100:
                    color = '#8B0000'  # Dark Red
                else:
                    color = cmap(norm(value))
                ax.add_patch(plt.Rectangle((i, j), 1, 1, color=color))

            # Set the limits and grid
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_xticks(np.arange(0, grid_size + 1, 1))
            ax.set_yticks(np.arange(0, grid_size + 1, 1))
            ax.grid(which='both', color='black')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)

            # Invert Y-axis to match typical matrix layout
            ax.invert_yaxis()

            # Add title
            if names is None:
                ax.set_title(chr(65 + idx), fontsize=20, loc='left')
            else:
                ax.set_title(name, fontsize=12)

            # Add arrows to show the policy
            for x in range(0, grid_size):
                for y in range(0, grid_size):
                    if grid[x, y] <= 0 and grid[x, y] != 0:
                        values = []
                        positions = []
                        if x > 0 and grid[x - 1, y] <= 0:  # left
                            values.append(grid[x - 1, y])
                        else:
                            if x<= 0: 
                                values.append(grid[x, y])
                            else:
                                values.append(grid[x, y]) if grid[x-1, y] == 50 else values.append(env.J["lava"])
                        positions.append((x - 1, y))
                        if x < grid_size - 1 and grid[x + 1, y] <= 0:  # right
                            values.append(grid[x + 1, y])
                        else:
                            if x >= grid_size - 1:
                                values.append(grid[x, y])
                            else:
                                values.append(grid[x, y]) if grid[x+1, y] == 50 else values.append(env.J["lava"])
                        positions.append((x + 1, y))
                        if y > 0 and grid[x, y - 1] <= 0:  # up
                            values.append(grid[x, y - 1])
                        else:
                            if y <= 0:
                                values.append(grid[x, y])
                            else:
                                values.append(grid[x, y]) if grid[x, y-1] == 50 else values.append(env.J["lava"])
                        positions.append((x, y - 1))
                        if y < grid_size - 1 and grid[x, y + 1] <= 0:  # down
                            values.append(grid[x, y + 1])
                        else:
                            if y >= grid_size - 1:
                                values.append(grid[x, y])
                            else:
                                values.append(grid[x, y]) if grid[x, y+1] == 50 else values.append(env.J["lava"])
                        positions.append((x, y + 1))

                        max_value = max(values)
                        max_indices = [index for index, value in enumerate(values) if value == max_value]
                        for index in max_indices:
                            target_x, target_y = positions[index]
                            dx, dy = target_x - x, target_y - y
                            ax.arrow(x + 0.5, y + 0.5, dx * 0.3, dy * 0.3,
                                    head_width=0.1, head_length=0.1, fc='#00008B', ec='#00008B', length_includes_head=True)  # Dark Blue

        # Remove empty subplots
        for j in range(idx + 1, len(axs)):
            fig.delaxes(axs[j])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(masked_grid)

        # Adjust layout to fit the colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.05, 0.015, 0.85])  # Adjusted to be thinner, taller, and closer to the plots
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Value Function')

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()


class Minigrid_MDP_Plotter(Plotter):

    def __init__(self, minigrid):
        super().__init__()
        self.minigrid = minigrid

    def minigrid_demo(self):
        print("Welcome to our Minigrid Interactive Demo")
        action = np.random.choice(self.minigrid.n_actions)
        a = input(f"       Select an Action from {self.minigrid.actions} (default {action}): ")
        while a != 'e':
            if a and a.isdigit():
                action = int(a)
                _, reward, terminated, truncated, info = self.minigrid.env.step(action)
                print("Minigrid Step")
                print("       Action:", action)
                print("  Observation:", self.minigrid.observation())
                print("       Reward:", reward)
                print("         Done:", "terminated" if terminated else "truncated" if truncated else "False", )
                print("         Info:", info)
                self.minigrid.render()
            else:
                print("Invalid action.")
            a = input(f"       Select an Action from {self.minigrid.actions} (default {action}, press 'e' to exit): ")

    def state_visited_policy(self, policy):
        state = self.minigrid.s0
        states_visited = np.zeros((self.minigrid.n_states))
        states_visited[state] = 1
        n_steps = 0
        while True:
            n_steps += 1
            next_state, _, done = self.minigrid.act(state, policy[state])
            states_visited[next_state] = 1
            if done:
                break
            else:
                state = next_state

            if n_steps > 100:
                break

        return states_visited

    def plot_greedy_policy(self, greedy_policy, estimated=False, print_policy=False):
        if print_policy: print(greedy_policy)
        states_visited = self.state_visited_policy(greedy_policy)
        plt.figure(figsize=(5, 2*self.minigrid.grid_size))
        plt.imshow(states_visited.reshape((self.minigrid.n_cells,4)), cmap="Greys")
        if estimated: plt.title("Estimated policy \n ( The states in black represent the states selected by the estimated policy)")
        else: plt.title("Optimal policy \n ( The states in black represent the states selected by the Optimal policy)")
        plt.xlabel("Orientation")
        plt.ylabel("Cell Position")
        y_labels = [f"{i//self.minigrid.grid_size + 1}x{i%self.minigrid.grid_size + 1}" for i in range(self.minigrid.n_cells)]
        x_labels = [f"{i}" for i in range(self.minigrid.n_orientations)]
        plt.yticks(range(self.minigrid.n_cells), y_labels)
        plt.xticks(range(self.minigrid.n_orientations), x_labels)
        plt.show()   

    def plot_value_function(self, Q, print_values=False, plot_values=False, file = "value_function.txt"):
        V = Q.max(axis=1)
        if print_values: 
            with open(file, "w") as f:
                for s in range(self.minigrid.n_states):
                    f.write("V({}): {}\n".format(self.minigrid.states[s], V[s]))
                    f.write("Q({}): {}\n".format(self.minigrid.states[s], Q[s]))

        if plot_values:

            plt.figure(figsize=(5, 2*self.minigrid.grid_size))
            im = plt.imshow(V.reshape((self.minigrid.n_cells,4)), vmin= V.min(), vmax=V.max())
            colorbar_ticks = np.linspace(V.max(), V.min(), 10)
            plt.colorbar(im, ticks=colorbar_ticks) 
            plt.title("V(s)")
            plt.xlabel("Orientation")
            plt.ylabel("Cell Position")
            y_labels = [f"{i//self.minigrid.grid_size + 1}x{i%self.minigrid.grid_size + 1}" for i in range(self.minigrid.n_cells)]
            x_labels = [f"{i}" for i in range(self.minigrid.n_orientations)]
            plt.yticks(range(self.minigrid.n_cells), y_labels)
            plt.xticks(range(self.minigrid.n_orientations), x_labels)
            plt.show()

    def plot_path(self, policy, start = 0, path='MDP_policy_path.gif'):
        self.minigrid.s0 = start if self.minigrid.s0 != start else self.minigrid.s0
        s = self.minigrid.reset()
        done = False
        with imageio.get_writer(path, mode='I', duration=0.2, loop=10) as writer:
            writer.append_data(self.minigrid.env.render())
            while not (done and self.minigrid.is_goal(s)):
                s, _, done = self.minigrid.step(s, policy[s]) if not self.minigrid.is_goal(s) else (start, 0, False)
                writer.append_data(self.minigrid.env.render())
        os.startfile(path)  # for windows

    def plot_greedy_policy_square(self, greedy_policy):
        states_visited = self.state_visited_policy(greedy_policy)
        _, axs = plt.subplots(self.minigrid.grid_size, self.minigrid.grid_size, figsize=(30, 80))

        # Calculate the number of orientations per cell
        num_orientations = self.minigrid.n_orientations

        # Iterate over each cell
        for row in range(self.minigrid.grid_size):
            for col in range(self.minigrid.grid_size):
                ax = axs[row, col]

                # Calculate the indices for the orientations of the current cell
                base_index = self.minigrid.state_to_index[(col + 1, row + 1, 0)]
                indices = [base_index + o for o in range(num_orientations)]

                # Create an empty grid for the current cell
                cell_grid = np.zeros((num_orientations, num_orientations))

                # Mark visited states within the cell
                for index in indices:
                    if states_visited[index] == 1:
                        orientation = index - base_index
                        cell_grid[orientation % num_orientations, orientation // num_orientations] = 1

                # Plot the grid for the current cell
                ax.imshow(cell_grid, cmap="Greys", origin='upper')

                # Set title and remove axes ticks
                ax.set_title(f"Cell ({col+1}, {row+1})")
                ax.axis("off")

        plt.suptitle("Greedy Policy\n(The states in black represent the states selected by the greedy policy)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
