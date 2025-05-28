"""
Monte Carlo agent for first-visit control in RL.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, actions, epsilon=0.1, gamma=1.0, seed=None):
        """
        Initialize agent.
        actions: list of possible actions.
        epsilon: exploration rate.
        gamma: discount factor.
        seed: random seed.
        """

        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(len(actions)))
        self.returns = defaultdict(lambda: [[] for _ in actions])
        self.episode_returns = []
        self.rng = np.random.default_rng(seed)

    def policy(self, state, method="epsilon", **kwargs):
        """
        Select action using specified policy.
        method: "epsilon", "greedy", "softmax", or "decay".
        """

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

    def update(self, episode):
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                self.returns[state][action].append(G)
                self.Q[state][action] = np.mean(self.returns[state][action])
                visited.add((state, action))

    def evaluate_policy(self, env, episodes=100, method="greedy", **kwargs):
        """
        Evaluate policy over several episodes.
        Returns: (mean, std) of rewards.
        """

        if not hasattr(env, 'reset') or not hasattr(env, 'step'):
            raise ValueError("Provided environment does not have the required Gym API.")

        total_rewards = []
        for _ in range(episodes):
            state_info = env.reset()
            state = state_info[0] if isinstance(state_info, tuple) else state_info
            done = False
            ep_reward = 0

            while not done:
                action = self.policy(state, method=method, **kwargs)
                step_result = env.step(action)

                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated

                else:
                    next_state, reward, done, _ = step_result
                    
                ep_reward += reward
                state = next_state

            total_rewards.append(ep_reward)
            self.episode_returns.append(ep_reward)

        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)

        print(f"Evaluation completed: {episodes} episodes")
        print(f"→ Average Return: {mean_reward:.2f} ± {std_reward:.2f}")

        return mean_reward, std_reward

    def get_policy(self):
        return {state: np.argmax(actions) for state, actions in self.Q.items()}

    def save(self, path):
        np.save(path, dict(self.Q))

    def load(self, path):
        data = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)), data)

    def plot_learning_curve(self, window=100, save_path=None):
        if not self.episode_returns:
            print("No episode return data to plot.")
            return
        
        if len(self.episode_returns) < window:
            print(f"Not enough data to compute moving average with window size {window}.")
            return
        
        returns = np.array(self.episode_returns)
        moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')

        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg, label=f"{window}-episode moving average")
        plt.title("Learning Curve: Episode Return Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        else:
            plt.show()

    def export_policy_csv(self, filename):
        if not self.Q:
            print("Warning: Q-table is empty. Nothing to export.")
            return
        
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["State", "BestAction"])
                count = 0

                for state, q_values in self.Q.items():
                    state_str = str(state)
                    best_action = int(np.argmax(q_values))
                    writer.writerow([state_str, best_action])
                    count += 1
                
            print(f"✅ Exported policy with {count} states to '{filename}'.")

        except Exception as e:
            print(f"❌ Failed to export policy: {e}")

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (f"MonteCarloAgent(actions={self.actions}, epsilon={self.epsilon}, "
                f"gamma={self.gamma}")