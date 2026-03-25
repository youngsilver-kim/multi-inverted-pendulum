# Multi Inverted Pendulum Control via Deep Reinforcement Learning

## Overview

This project addresses the control problem of a multi-link inverted pendulum system (3–5 links) using deep reinforcement learning.
Unlike the classical single pendulum, the multi-link system exhibits highly nonlinear, coupled dynamics, making it significantly more challenging to stabilize.

The goal is to learn a control policy that keeps all pendulum links upright while maintaining the cart position within a stable range.

---

## Environment

A custom environment is implemented using Gymnasium.

### State Space

The system state consists of:

* Cart position and velocity
* Angular positions and velocities of each pendulum link

Example (3-link system):
x, θ₁, θ₂, θ₃, ẋ, θ̇₁, θ̇₂, θ̇₃

### Action Space

* Continuous force applied to the cart

### Dynamics

* Nonlinear system modeled using coupled equations of motion
* Numerical solution via linear system solver
* Time integration using Euler method

---

## Method

We adopt Proximal Policy Optimization (PPO) for continuous control.

* Framework: PyTorch
* Library: Stable-Baselines3
* Policy: MLP

### Training Strategy

* Reward shaping based on:

  * Pendulum angle deviation
  * Cart position error
  * Velocity penalties
* Early termination when instability thresholds are exceeded

---

## Implementation Details

Key components of the implementation include:

* Custom Gym environment for multi-link pendulum control 
* Numerical stabilization in dynamics computation (matrix regularization)
* Continuous control via PPO
* Real-time visualization using Pygame
* Video recording using MoviePy

---

## Results

The trained agent is capable of:

* Stabilizing multiple pendulum links simultaneously
* Maintaining upright equilibrium under nonlinear dynamics
* Generating smooth control signals


---


---

## Installation

```bash
pip install gymnasium stable-baselines3 torch pygame moviepy
```

---

## Usage

### Train the agent

```bash
python train.py
```

### Run simulation

The script automatically:

* trains the model (if not available)
* runs evaluation
* saves a video of the result

---

## Future Work

* Extension to 5-link inverted pendulum
* Comparison with SAC / TD3
* Integration with classical control (LQR)
* Improved physics modeling and numerical integration

---

## Keywords

Reinforcement Learning, PPO, Control Systems, Nonlinear Dynamics, Inverted Pendulum
