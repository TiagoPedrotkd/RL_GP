import numpy as np
import math

class Discretizer:
    def __init__(self, angle_buckets=25, vel_buckets=25, action_low=-2.0, action_high=2.0, n_actions=5):
        self.angle_buckets = angle_buckets
        self.vel_buckets = vel_buckets
        self.action_low = action_low
        self.action_high = action_high
        self.n_actions = n_actions
        self.actions = np.linspace(action_low, action_high, n_actions)
        self.window_angle = (2 * math.pi) / angle_buckets
        self.window_vel = 16.0 / vel_buckets

    def discretise_state(self, observation):
        if isinstance(observation, np.ndarray):
            obs = observation.tolist()
        else:
            obs = observation

        if isinstance(obs, (tuple, list)) and len(obs) == 2 and all(isinstance(x, int) for x in obs):
            return tuple(obs)

        if isinstance(obs, (tuple, list)) and len(obs) == 3:
            cos_th, sin_th, th_dot = obs
            theta = math.atan2(sin_th, cos_th)
            angle_bin = int((theta + math.pi) // self.window_angle)
            vel_bin = int((th_dot + 8.0) // self.window_vel)
            angle_bin = np.clip(angle_bin, 0, self.angle_buckets - 1)
            vel_bin = np.clip(vel_bin, 0, self.vel_buckets - 1)
            return (angle_bin, vel_bin)
        else:
            raise ValueError(f"discretise_state: observation must have 3 elements (cos, sin, th_dot) or 2 ints (discretized), got {obs}")

    def initialize_q_table(self):
        shape = [self.angle_buckets, self.vel_buckets, self.n_actions]
        return np.zeros(shape, dtype=np.float32)

    def get_action(self, action_idx):
        return self.actions[action_idx]
