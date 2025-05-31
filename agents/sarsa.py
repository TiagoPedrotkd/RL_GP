"""
SARSA agent for on-policy temporal difference learning.
"""

import numpy as np
from collections import defaultdict

class SARSAAgent:
    def __init__(self, actions, alpha=0.1, gamma=1.0, epsilon=0.1, policy=None, seed=None):
        """
        Initialize agent.
        actions: list of possible actions.
        alpha: learning rate.
        gamma: discount factor.
        epsilon: exploration rate.
        policy: policy selection method ("epsilon", "greedy", "softmax", "decay").
        seed: random seed.
        """

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_method = policy or "epsilon"
        self.Q = defaultdict(lambda: np.zeros(len(actions)))
        self.rng = np.random.default_rng(seed)

    def policy(self, state, method=None, **kwargs):
        """
        Select action using specified policy.
        method: "epsilon", "greedy", "softmax", or "decay".
        If method is None, uses self.policy_method.
        """

        method = method or self.policy_method

        if method == "greedy":
            return self.policy_greedy(state)
        
        elif method == "softmax":
            temperature = kwargs.get("temperature", 1.0)
            return self.policy_softmax(state, temperature)
        
        elif method == "decay":
            episode = kwargs.get("episode", 0)
            decay_rate = kwargs.get("decay_rate", 0.0001)
            return self.policy_decay_epsilon(state, episode, decay_rate)
        
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        
        return np.argmax(self.Q[state])

    def policy_greedy(self, state):
        return np.argmax(self.Q[state])

    def policy_softmax(self, state, temperature=1.0):

        q = self.Q[state]
        exp_q = np.exp((q - np.max(q)) / temperature)
        probs = exp_q / np.sum(exp_q)

        return self.rng.choice(self.actions, p=probs)

    def policy_decay_epsilon(self, state, episode, decay_rate=0.0001):
        eps = max(0.01, self.epsilon * np.exp(-decay_rate * episode))
        
        if self.rng.random() < eps:
            return self.rng.choice(self.actions)
        
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action):
        target = reward + self.gamma * self.Q[next_state][next_action]

        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def get_policy(self):
        return {s: np.argmax(a) for s, a in self.Q.items()}

    def save(self, path):
        np.save(path, dict(self.Q))

    def load(self, path):
        data = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)), data)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (f"SARSAAgent(actions={self.actions}, alpha={self.alpha}, "
                f"gamma={self.gamma}, epsilon={self.epsilon}, policy={self.policy_method})")