import numpy as np

class RandomAgent:
    def __init__(self, actions, seed=None):
        self.actions = actions
        self.rng = np.random.default_rng(seed)
        self.episode_returns = []

    def policy(self, state, **kwargs):
        return self.rng.choice(self.actions)

    def update(self, *args, **kwargs):
        pass  # Random agent does not learn

    def save(self, path):
        pass  # Nothing to save

    def load(self, path):
        pass  # Nothing to load

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return f"RandomAgent(actions={self.actions})"
