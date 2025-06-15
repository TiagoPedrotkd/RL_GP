"""
Módulo utilitário para criação do ambiente Blackjack-v1 do Gymnasium.
"""

import gymnasium as gym

class BlackjackEnvWrapper:
    """
    Wrapper para o ambiente Blackjack-v1, compatível com Gymnasium, com métodos reset, step, render e close.
    Permite definir seed para reprodutibilidade.
    """
    def __init__(self, env_name="Blackjack-v1", seed=None):
        self.env = gym.make(env_name)
        self.seed = seed
        self._set_seed(seed)

    def _set_seed(self, seed):
        if seed is not None:
            self.env.reset(seed=seed)
            if hasattr(self.env.action_space, "seed"):
                self.env.action_space.seed(seed)
            if hasattr(self.env.observation_space, "seed"):
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

def make_env(env_name="Blackjack-v1", seed=None):
    """
    Cria e retorna um wrapper do ambiente Blackjack-v1 com seed opcional.
    """
    return BlackjackEnvWrapper(env_name=env_name, seed=seed)
