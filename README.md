# Multi Inverted Pendulum Control with Reinforcement Learning

## Overview

This project explores controlling multi-link inverted pendulum systems (3, 4, and 5 links) using reinforcement learning.
As the number of links increases, the system becomes significantly more unstable due to nonlinear and coupled dynamics.
The goal is to learn a policy that keeps all links balanced upright while controlling the cart position.

---

## Motivation

The inverted pendulum is a classic control problem, but most implementations focus on single or double pendulums.
This project extends the problem to multiple links, where traditional control methods become difficult to apply.

By using reinforcement learning, the system can learn control strategies directly from interaction without requiring an explicit analytical solution.

---

## Environment

Custom environments are implemented using Gymnasium.

Each environment includes:

* Continuous state space (position, angles, velocities)
* Continuous action space (force applied to the cart)
* Nonlinear dynamics solved numerically
* Termination based on angle and position limits

State representation (example for 3-link):

x, θ₁, θ₂, θ₃, x_dot, θ₁_dot, θ₂_dot, θ₃_dot

---

## Methods

Two reinforcement learning algorithms are used:

### PPO (Proximal Policy Optimization)

* Stable and widely used policy gradient method
* Performs well on continuous control tasks

### SAC (Soft Actor-Critic)

* Off-policy algorithm
* Better exploration due to entropy maximization
* Used for comparison

---

## Experiments

We trained agents on three different environments:

* Triple inverted pendulum (3 links)
* Quadruple inverted pendulum (4 links)
* Quintuple inverted pendulum (5 links)

Training difficulty increases rapidly with the number of links.

---

## Results

### Performance by Number of Links

| System | Difficulty | Stability         |
| ------ | ---------- | ----------------- |
| 3-link | Low        | Stable            |
| 4-link | Medium     | Partially stable  |
| 5-link | High       | Hard to stabilize |

---

### PPO vs SAC

| Algorithm | Characteristics                        |
| --------- | -------------------------------------- |
| PPO       | More stable training                   |
| SAC       | Better exploration but less consistent |

In this project, PPO generally showed more reliable performance for multi-link control.

---

## Visualization

### Simulation Result (3-link)

![result](results/v1_triple/triple_pendulum_control.mp4)

---

## Project Structure

```
env/        # custom environments
train/      # training scripts
models/     # trained models
logs/       # training logs
results/    # simulation outputs
```

---

## How to Run

Install dependencies:

pip install gymnasium stable-baselines3 torch pygame moviepy

Run training:

python train/train_triple.py
python train/train_quadruple.py
python train/train_quintuple.py

---

## Key Points

* Custom multi-link physics environments
* Reinforcement learning-based control
* Comparison across different system complexities
* PPO vs SAC comparison

---

## Future Work

* Improve stability for 5-link system
* Compare with classical control methods (LQR)
* Add performance metrics and plots
* Optimize reward design

---

## Author

AI undergraduate student focusing on reinforcement learning and control systems.
