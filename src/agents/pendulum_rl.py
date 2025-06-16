import numpy as np
from src.agents.q_learning import DiscretizedQLearningAgent
from src.agents.sarsa import DiscretizedSARSAAgent
from src.agents.discretizer import Discretizer

class PendulumRL:
    """
    Classe utilitária para treino RL no Pendulum-v1 com discretização.
    """
    def __init__(self, angle_buckets=25, vel_buckets=25, action_low=-2.0, action_high=2.0, n_actions=5):
        self.discretizer = Discretizer(angle_buckets, vel_buckets, action_low, action_high, n_actions)

    def train(self, env, method="qlearning", num_iterations=10000, alpha=0.1, gamma=0.99, epsilon=0.1, policy="epsilon", verbose=True):
        agent = self._create_agent(method, alpha, gamma, epsilon, policy)
        episode_returns = []
        for ep in range(num_iterations):
            obs = self._reset_env(env)
            self.discretizer.discretise_state(obs)
            action_idx = agent.policy(obs)
            done = False
            total_reward = 0
            while not done:
                action = self.discretizer.get_action(action_idx)
                next_obs, reward, done = self._step_env(env, action)
                self.discretizer.discretise_state(next_obs)
                obs, action_idx = self._update_agent(agent, method, obs, action_idx, reward, next_obs)
                total_reward += reward
            episode_returns.append(total_reward)
            if verbose and (ep+1) % 1000 == 0:
                print(f"Episode {ep+1}/{num_iterations} | Return: {total_reward:.2f}")
        agent.episode_returns = episode_returns
        return agent.Q, {"Return": np.array(episode_returns)}

    def _create_agent(self, method, alpha, gamma, epsilon, policy):
        method = method.lower()
        if method == "qlearning":
            return DiscretizedQLearningAgent(self.discretizer, alpha=alpha, gamma=gamma, epsilon=epsilon, policy=policy)
        elif method == "sarsa":
            return DiscretizedSARSAAgent(self.discretizer, alpha=alpha, gamma=gamma, epsilon=epsilon, policy=policy)
        else:
            raise ValueError("Método não suportado: use 'qlearning' ou 'sarsa'.")

    def _reset_env(self, env):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            return reset_result[0]
        return reset_result

    def _step_env(self, env, action):
        step_result = env.step([action])
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            next_obs, reward, done, _ = step_result
        else:
            raise ValueError("Formato inesperado no retorno do ambiente Pendulum.")
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        return next_obs, reward, done

    def _update_agent(self, agent, method, obs, action_idx, reward, next_obs):
        if method.lower() == "qlearning":
            agent.update(obs, action_idx, reward, next_obs)
            next_action_idx = agent.policy(next_obs)
        else:
            next_action_idx = agent.policy(next_obs)
            agent.update(obs, action_idx, reward, next_obs, next_action_idx)
        return next_obs, next_action_idx
