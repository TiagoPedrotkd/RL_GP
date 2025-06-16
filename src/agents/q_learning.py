import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent para ambientes com espaço de ações e estados discretos (ex: Blackjack).
    """
    def __init__(self, actions, alpha=0.1, gamma=1.0, epsilon=0.1, policy=None, seed=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_method = policy or "epsilon"
        self.Q = defaultdict(lambda: np.zeros(len(actions)))
        self.rng = np.random.default_rng(seed)

    def policy(self, state, method=None, **kwargs):
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

    def update(self, state, action, reward, next_state):
        max_next_q = np.max(self.Q[next_state])
        target = reward + self.gamma * max_next_q
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
        return (f"QLearningAgent(actions={self.actions}, alpha={self.alpha}, "
                f"gamma={self.gamma}, epsilon={self.epsilon}, policy={self.policy_method}")


class DiscretizedQLearningAgent:
    """
    Q-Learning agent para ambientes contínuos discretizados (ex: Pendulum).
    Usa uma Q-table numpy para (angle_bin, vel_bin, action_idx).
    """
    def __init__(self, discretizer, alpha=0.1, gamma=0.99, epsilon=0.1, policy="epsilon", seed=None):
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_method = policy
        self.Q = self.discretizer.initialize_q_table()
        self.rng = np.random.default_rng(seed)
        self.actions = self.discretizer.actions

    def policy(self, obs, method=None, **kwargs):
        method = method or self.policy_method
        discrete_state = self.discretizer.discretise_state(obs)
        if method == "greedy":
            return int(np.argmax(self.Q[discrete_state]))
        elif method == "softmax":
            temperature = kwargs.get("temperature", 1.0)
            q = self.Q[discrete_state]
            exp_q = np.exp((q - np.max(q)) / temperature)
            probs = exp_q / np.sum(exp_q)
            return int(self.rng.choice(len(self.actions), p=probs))
        elif method == "decay":
            episode = kwargs.get("episode", 0)
            decay_rate = kwargs.get("decay_rate", 0.0001)
            eps = max(0.01, self.epsilon * np.exp(-decay_rate * episode))
            if self.rng.random() < eps:
                return int(self.rng.integers(len(self.actions)))
            return int(np.argmax(self.Q[discrete_state]))
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self.Q[discrete_state]))

    def update(self, obs, action_idx, reward, next_obs):
        state = self.discretizer.discretise_state(obs)
        next_state = self.discretizer.discretise_state(next_obs)
        max_next_q = np.max(self.Q[next_state])
        target = reward + self.gamma * max_next_q
        self.Q[state][action_idx] += self.alpha * (target - self.Q[state][action_idx])

    def get_policy(self):
        policy = {}
        for angle_bin in range(self.discretizer.angle_buckets):
            for vel_bin in range(self.discretizer.vel_buckets):
                state = (angle_bin, vel_bin)
                policy[state] = int(np.argmax(self.Q[state]))
        return policy

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path, allow_pickle=True)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (f"DiscretizedQLearningAgent(alpha={self.alpha}, gamma={self.gamma}, "
                f"epsilon={self.epsilon}, policy={self.policy_method}, "
                f"buckets=({self.discretizer.angle_buckets},{self.discretizer.vel_buckets}), "
                f"n_actions={self.discretizer.n_actions})")