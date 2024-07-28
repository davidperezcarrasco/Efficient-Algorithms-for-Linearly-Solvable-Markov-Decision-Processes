# Efficient Algorithms for Linearly-Solvable Markov Decision Processes

As part of my Bachelor's Thesis, I conducted in-depth research on reinforcement learning (RL) algorithms, addressing their scalability and efficiency in large and complex domains. This thesis explores the application of Linearly Solvable Markov Decision Processes (LMDPs) to tackle these challenges. A significant contribution of my work is the development of novel embedding techniques for creating precise and exact mappings between traditional MDPs and LMDPs. These techniques improve the approximation precision by 99.23% over the approach by Todorov, enabling robust and reliable comparisons across the two frameworks, regardless of the problem's definition or the nature of its dynamics.

This research evaluates and benchmarks the performance of traditional RL models against algorithms leveraging LMDPs, such as Z-learning, within an adaptable reinforcement learning framework. By implementing scalable and efficient versions of these algorithms, I provide a comprehensive comparison that highlights the advantages of LMDP-based approaches. Additionally, the thesis delves into various factors that enhance the decision-making capabilities of RL agents, such as algorithm design and exploration strategies, demonstrating the superiority of the LMDP framework in multiple settings. Moreover, I developed an RL simulator that provides a generalized implementation of MDP and LMDP frameworks. This simulator supports optimized core methods and seamless integration with repositories like Minigrid and Gymnasium, allowing researchers to easily extend and apply it to diverse problem definitions without the need to build custom solutions.

## Embedding Stochastic MDP into LMDPs

![Embedding of stochastic MDP into LMDP](assets/img/lmdps/stochastic-mdp-embedding.png)
*Embedding of stochastic MDP into LMDP implementation*

### Embedding Deterministic MDPs into LMDPs

![Embedding of deterministic MDP into LMDP](assets/img/lmdps/deterministic-mdp-embedding-spa.png)
*Embedding of deterministic MDP into LMDP implementation through SPA method*

$$
\begin{aligned}
\min_{K \in \mathbb{R}} \quad \quad & \left\| \mathbf{v}_K - \mathbf{v}^* \right\|^2 \\
\text{subject to} \quad \quad & \mathbf{\mathcal{R}} = K \cdot \mathbf{\hat{\mathcal{R}}}, \\
& G_{ii} = e^{\mathcal{R}(s_i)/\lambda}, \quad \forall i \in \mathbb{N} \cap [1, |\mathcal{S}|], \\
& \mathbf{z}_K = G\mathcal{P}\mathbf{z}_K, \\
& \mathbf{v}_K = \lambda \log{\mathbf{z}_K}, \\
& \mathbf{v}^* = \max_{a} \left[ \tilde{\mathbf{R}}_a + \gamma \mathbf{\tilde{P}}_a \mathbf{v}^* \right].
\end{aligned}
$$

![Embedding of deterministic MDP into LMDP through TS method](assets/img/lmdps/deterministic-mdp-embedding-ts.png)
*Embedding of deterministic MDP into LMDP implementation through TS method*

![Deterministic MDP Embedding Methodologies Comparison](assets/img/lmdps/deterministic-mdp-embedding-scatter.png)
*Deterministic MDP Embedding Approximations Comparison*

![Deterministic MDP Embedding Methodologies Comparison in large settings](assets/img/lmdps/deterministic-mdp-embedding-scatter-large.png)
*Deterministic MDP Embedding Approximations Comparison in large settings*

![Deterministic MDP Embedding Approximations MSE Comparison by problem size](assets/img/lmdps/deterministic-mdp-embedding-mse.png)
*Deterministic MDP Embedding Approximations MSE Comparison by problem size*

## Experimentation
### Exploration Strategies in Traditional MDPs

![Q-learning convergence and approximation MSE by exploration decay](assets/img/lmdps/q-learning-epsilon-decay.png)
*Q-learning convergence and approximation MSE by exploration decay*

### Comparative Analysis of Z-learning and Q-learning
#### Approximation Error and Episodic Reward

![Comparison of approximation error (top) and episodic reward (bottom) between Q-learning and Z-learning](assets/img/lmdps/zvsq-rew-err.png)
*Comparison of approximation error (top) and episodic reward (bottom) between Q-learning and Z-learning.*

#### Value Function and Policy Approximations

![Value function and policy approximations for Q-learning and Z-learning in a Minigrid multi-room domain](assets/img/lmdps/zvsq-val-multiroom.png)
*Value function and policy approximations for Q-learning and Z-learning in a Minigrid multi-room domain*

![Value function and policy approximations for Q-learning and Z-learning in a Minigrid hill-cliff domain](assets/img/lmdps/zvsq-val-hill.png)
*Value function and policy approximations for Q-learning and Z-learning in a Minigrid hill-cliff domain.*

### Agent Solutions

![Grid World Maze with Q-learning](assets/video/lmdps/small-maze-lmdp.mp4)
*Grid World Maze with Q-learning*

![Large Grid World Maze with Q-learning](assets/video/lmdps/large-maze-mdp.mp4)
*Large Grid World Maze with Q-learning*

![Grid World Hill-Cliff with Q-learning](assets/video/lmdps/hill.mp4)
*Grid World Hill-Cliff with Q-learning*

![Grid World Multi-Room with Z-learning](assets/video/lmdps/multi-room-lmdp.mp4)
*Grid World Multi-Room with Z-learning*

## Requirements

The project requires the following Python libraries:

- `gym==0.26.2`
- `imageio==2.31.4`
- `matplotlib==3.8.0`
- `minigrid==2.3.1`
- `numpy==1.26.4`
- `scipy==1.12.0`

You can install these requirements using pip:

```bash
pip install -r requirements.txt

**Link to project webpage: https://davidperezcarrasco.github.io/projects/1_project/**
