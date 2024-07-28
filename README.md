# Efficient Algorithms for Linearly-Solvable Markov Decision Processes

As part of my Bachelor's Thesis, I conducted in-depth research on reinforcement learning (RL) algorithms, addressing their scalability and efficiency in large and complex domains. This thesis explores the application of Linearly Solvable Markov Decision Processes (LMDPs) to tackle these challenges. A significant contribution of my work is the development of novel embedding techniques for creating precise and exact mappings between traditional MDPs and LMDPs. These techniques improve the approximation precision by 99.23% over the approach by Todorov, enabling robust and reliable comparisons across the two frameworks, regardless of the problem's definition or the nature of its dynamics.

This research evaluates and benchmarks the performance of traditional RL models against algorithms leveraging LMDPs, such as Z-learning, within an adaptable reinforcement learning framework. By implementing scalable and efficient versions of these algorithms, I provide a comprehensive comparison that highlights the advantages of LMDP-based approaches. Additionally, the thesis delves into various factors that enhance the decision-making capabilities of RL agents, such as algorithm design and exploration strategies, demonstrating the superiority of the LMDP framework in multiple settings. Moreover, I developed an RL simulator that provides a generalized implementation of MDP and LMDP frameworks. This simulator supports optimized core methods and seamless integration with repositories like Minigrid and Gymnasium, allowing researchers to easily extend and apply it to diverse problem definitions without the need to build custom solutions.

## Embedding Stochastic MDP into LMDPs
<div class="row justify-content-sm-center">
    <div class="col-sm-9">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/stochastic-mdp-embedding.png" title="Embedding of stochastic MDP into LMDP" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Embedding of stochastic MDP into LMDP implementation
</div>

### Embedding Deterministic MDPs into LMDPs

<div class="row justify-content-sm-center">
    <div class="col-sm-9">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/deterministic-mdp-embedding-spa.png" title="Embedding of deterministic MDP into LMDP" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Embedding of deterministic MDP into LMDP implementation through SPA method
</div>

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

<div class="row justify-content-sm-center">
    <div class="col-sm-12">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/deterministic-mdp-embedding-ts.png" title="Embedding of deterministic MDP into LMDP" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Embedding of deterministic MDP into LMDP implementation through TS method
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-12">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/deterministic-mdp-embedding-scatter.png" title="Deterministic MDP Embedding Methodologies Comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Deterministic MDP Embedding Approximations Comparison
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-12">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/deterministic-mdp-embedding-scatter-large.png" title="PDeterministic MDP Embedding Methodologies Comparison in large settings" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption text-center">
    Deterministic MDP Embedding Approximations Comparison in large settings
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-12">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/deterministic-mdp-embedding-mse" title="Deterministic MDP Embedding Methodologies MSE comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Deterministic MDP Embedding Approximations MSE Comparison by problem size
</div>

## Experimentation
### Exploration Strategies in Traditional MDPs

<div class="row justify-content-sm-center">
    <div class="col-sm-9">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/q-learning-epsilon-decay.png" title="Q-learning convergence and approximation MSE by exploration decay" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Q-learning convergence and approximation MSE by exploration decay
</div>

### Comparative Analysis of Z-learning and Q-learning
#### Approximation Error and Episodic Reward

<div class="row justify-content-sm-center">
    <div class="col-sm-9">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/zvsq-rew-err.png" title="Comparison of approximation error (top) and episodic reward (bottom) between Q-learning and Z-learning" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of approximation error (top) and episodic reward (bottom) between Q-learning and Z-learning.
</div>

#### Value Function and Policy Approximations

<div class="row justify-content-sm-center">
    <div class="col-sm-12">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/zvsq-val-multiroom.png" title="Value function and policy approximations for Q-learning and Z-learning" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Value function and policy approximations for Q-learning and Z-learning in a Minigrid multi-room domain
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-12">
        {% include figure.liquid loading="eager" path="assets/img/lmdps/zvsq-val-hill.png" title="Value function and policy approximations for Q-learning and Z-learning" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Value function and policy approximations for Q-learning and Z-learning in a Minigrid hill-cliff domain.
</div>

#### Agents Solutions

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.liquid loading="eager" path="assets/video/lmdps/small-maze-lmdp.mp4" title="Grid World Maze with Q-learning" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
        <div class="caption">
            Grid World Maze with Q-learning
        </div>
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.liquid loading="eager" path="assets/video/lmdps/large-maze-mdp.mp4" title="Large Grid World Maze with Q-learning" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
        <div class="caption">
            Large Grid World Maze with Q-learning
        </div>
    </div>
</div>


<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.liquid loading="eager" path="assets/video/lmdps/hill.mp4" title="Grid World Hill-Cliff with Q-learning" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
        <div class="caption">
            Grid World Hill-Cliff with Q-learning
        </div>
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.liquid loading="eager" path="assets/video/lmdps/multi-room-lmdp.mp4" title="Grid World Multi-Room with Z-learning" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
        <div class="caption">
            Grid World Multi-Room with Z-learning
        </div>
    </div>
</div>

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
