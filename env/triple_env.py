import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class TripleInvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self):
        super().__init__()

        self.m_c = 1.0
        self.m = np.array([0.5, 0.5, 0.5])
        self.l = np.array([0.5, 0.5, 0.5])
        self.g = 9.81
        self.tau = 0.02
        self.max_force = 30.0

        high = np.array(
            [5.0, np.pi, np.pi, np.pi,
             np.finfo(np.float32).max, np.finfo(np.float32).max,
             np.finfo(np.float32).max, np.finfo(np.float32).max],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(8,)).astype(np.float32)
        self.state[1:4] = self.np_random.uniform(low=-0.2, high=0.2, size=(3,))
        return self.state, {}

    def _get_dynamics(self, state, force):
        x, t1, t2, t3, xd, t1d, t2d, t3d = state
        c1, s1 = math.cos(t1), math.sin(t1)

        l1, l2, l3 = self.l
        m1, m2, m3 = self.m
        mc, g = self.m_c, self.g

        M = np.zeros((4, 4))
        RHS = np.zeros((4, 1))

        M[0, 0] = mc + m1 + m2 + m3 + 1e-2
        M[0, 1] = (m1 + m2 + m3) * l1 * c1
        M[1, 0] = M[0, 1]
        M[1, 1] = (m1 + m2 + m3) * l1**2 + 1e-2

        RHS[0] = force[0] + (m1 + m2 + m3) * l1 * s1 * t1d**2
        RHS[1] = - (m1 + m2 + m3) * l1 * g * s1

        try:
            q_ddot = np.linalg.solve(M, RHS)
        except:
            q_ddot = np.zeros((4, 1))

        return q_ddot.flatten()

    def step(self, action):
        force = np.clip(action, -self.max_force, self.max_force)

        qd = self.state[4:]
        qdd = self._get_dynamics(self.state, force)
        self.state = self.state + self.tau * np.concatenate((qd, qdd))

        x, t1, t2, t3 = self.state[:4]

        reward = 100 - (t1**2 + t2**2 + t3**2) * 50 - x**2
        reward = np.clip(reward, -10, 100)

        terminated = bool(
            (abs(self.state[1:4]) > np.deg2rad(90)).any() or
            abs(x) > 4.8
        )

        return self.state.astype(np.float32), float(reward), terminated, False, {}
