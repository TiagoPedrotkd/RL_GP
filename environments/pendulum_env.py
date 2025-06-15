import gymnasium as gym
import numpy as np

class PendulumEnvWrapper:
    """
    Wrapper para o ambiente Pendulum-v1, com reset e seed compatíveis com Gymnasium.
    """
    def __init__(self, seed=None):
        self.env = gym.make("Pendulum-v1")
        if seed is not None:
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

def make_pendulum_env(seed=None):
    return PendulumEnvWrapper(seed=seed)
