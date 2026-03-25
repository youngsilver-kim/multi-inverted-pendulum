import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class QuadrupleInvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self):
        super().__init__()

        self.m_c = 1.0
        self.m = np.array([0.5]*4)
        self.l = np.array([0.5]*4)
        self.g = 9.81

        self.tau = 0.01
        self.max_force = 50.0
        self.angle_weight = 100.0

        high = np.ones(10, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.05, 0.05, size=(10,))
        return self.state.astype(np.float32), {}

    def _get_dynamics(self, state, force):
        x, t1, t2, t3, t4, xd, t1d, t2d, t3d, t4d = state
        c1, s1 = math.cos(t1), math.sin(t1)

        M = np.eye(5) * 1e-2
        M[0, 0] += sum(self.m) + self.m_c
        M[1, 1] += sum(self.m)

        RHS = np.zeros((5, 1))
        RHS[0] = force[0]
        RHS[1] = -self.g * s1

        try:
            return np.linalg.solve(M, RHS).flatten()
        except:
            return np.zeros(5)

    def step(self, action):
        force = np.clip(action, -self.max_force, self.max_force)

        qd = self.state[5:]
        qdd = self._get_dynamics(self.state, force)
        self.state = self.state + self.tau * np.concatenate((qd, qdd))

        x = self.state[0]
        angles = self.state[1:5]

        reward = 100 - self.angle_weight * np.sum(angles**2) - x**2
        reward = np.clip(reward, -10, 100)

        terminated = bool((abs(angles) > np.deg2rad(90)).any() or abs(x) > 4.8)

        return self.state.astype(np.float32), float(reward), terminated, False, {}
