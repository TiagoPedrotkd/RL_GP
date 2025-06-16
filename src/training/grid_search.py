import itertools
import numpy as np
from src.training.train_utils import RLTrainingUtils
from src.agents.pendulum_rl import PendulumRL

class BlackjackGridSearch:
    """
    Classe utilitária para grid search de hiperparâmetros no ambiente Blackjack-v1.
    Agora suporta múltiplos valores de 'episodes' no grid do YAML.
    """
    def __init__(self, config, n_seeds=3, verbose=True):
        self.config = config
        self.n_seeds = n_seeds
        self.verbose = verbose

    def _prepare_param_grid(self):
        grid = self.config.get("grid", {})
        agent_type = self.config["agent"]["type"]
        actions = [0, 1]

        episodes_list = grid.get("episodes", [self.config["training"]["episodes"]])
        if "alpha" in grid or "alpha" in self.config["agent"]:
            alpha_list = grid.get("alpha", [self.config["agent"].get("alpha", 0.1)])
        else:
            alpha_list = [None]
        gamma_list = grid.get("gamma", [self.config["agent"].get("gamma", 1.0)])
        epsilon_list = grid.get("epsilon", [self.config["agent"].get("epsilon", 0.1)])
        policies = grid.get("policy", [self.config["agent"].get("policy", "epsilon")])

        param_grid = list(itertools.product(alpha_list, gamma_list, epsilon_list, policies, episodes_list))
        return agent_type, actions, param_grid

    def _run_training_for_params(self, agent_type, actions, params, episodes):
        scores = []
        agent = None
        for seed in range(self.n_seeds):
            local_config = dict(self.config)
            local_config["seed"] = self.config.get("seed", 42) + seed

            if "training" not in local_config:
                local_config["training"] = {}
            local_config["training"]["episodes"] = episodes
            mean_return, agent, _ = RLTrainingUtils.run_training(
                agent_type=agent_type,
                actions=actions,
                params=params,
                config=local_config
            )
            scores.append(mean_return)
            print(f"Seed {seed} | Params: {params} | Episodes: {episodes} | Mean Return: {mean_return:.3f}")
        return scores, agent

    def _update_best_agent(self, mean_score, params, episodes, agent, best_score, best_config, best_agent):
        import logging
        if mean_score > best_score:
            best_score = mean_score
            best_config = dict(params, episodes=episodes)
            best_agent = agent
            best_path = "output/best_qtable.npy"
            best_agent.save(best_path)
            if self.verbose:
                print(f"  * Novo melhor agente salvo em: {best_path}")
            logging.info(f"Novo melhor agente salvo em: {best_path}")
        return best_score, best_config, best_agent

    def run(self):
        import logging
        agent_type, actions, param_grid = self._prepare_param_grid()
        best_score = float('-inf')
        best_config = None
        best_agent = None
        results = []

        print("\n========== Iniciando Grid Search Blackjack ==========")
        for idx, (alpha, gamma, epsilon, policy, episodes) in enumerate(param_grid, 1):
            params = {}
            if alpha is not None:
                params["alpha"] = alpha
            params["gamma"] = gamma
            params["epsilon"] = epsilon
            params["policy"] = policy

            msg = f"[{idx}/{len(param_grid)}] Parâmetros: {params} | episodes: {episodes}"
            print(msg)
            logging.info(msg)

            scores, agent = self._run_training_for_params(agent_type, actions, params, episodes)
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            results.append((dict(params, episodes=episodes), mean_score, std_score, scores))
            if self.verbose:
                print(f"Params: {params} | Episodes: {episodes} | Média: {mean_score:.3f} | Std: {std_score:.3f}")

            best_score, best_config, best_agent = self._update_best_agent(
                mean_score, params, episodes, agent, best_score, best_config, best_agent
            )

        print("\n========== Grid Search Blackjack Finalizado ==========")
        print(f"Melhor configuração: {best_config}")
        print(f"Melhor score médio das seeds: {best_score:.4f}")
        logging.info("\nMelhor configuração encontrada:")
        logging.info(f"{best_config}")
        logging.info(f"Score: {best_score:.4f}")

        return results

class PendulumGridSearch:
    """
    Classe utilitária para grid search de hiperparâmetros no ambiente Pendulum-v1.
    """
    def __init__(self, env_config, param_grid, n_seeds=3, method="qlearning", verbose=True):
        self.env_config = env_config
        self.param_grid = param_grid
        self.n_seeds = n_seeds
        self.method = method
        self.verbose = verbose

    def run(self):

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        results = []

        print("\n========== Iniciando Grid Search Pendulum ==========")
        for idx, param_tuple in enumerate(itertools.product(*values), 1):
            params = dict(zip(keys, param_tuple))
            scores = []
            for seed in range(self.n_seeds):
                env = self.env_config["make_env"](seed=seed)
                rl = PendulumRL(
                    angle_buckets=self.env_config.get("angle_buckets", 25),
                    vel_buckets=self.env_config.get("vel_buckets", 25),
                    action_low=self.env_config.get("action_low", -2.0),
                    action_high=self.env_config.get("action_high", 2.0),
                    n_actions=self.env_config.get("n_actions", 5)
                )
                _, metrics = rl.train(
                    env,
                    method=self.method,
                    num_iterations=params.get("episodes", 10000),
                    alpha=params.get("alpha", 0.1),
                    gamma=params.get("gamma", 0.99),
                    verbose=False
                )
                mean_return = metrics["Return"].mean()
                scores.append(mean_return)
                if self.verbose:
                    print(f"Seed {seed} | Params: {params} | Mean Return: {mean_return:.3f}")
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            results.append((params, mean_score, std_score, scores))
            if self.verbose:
                print(f"Params: {params} | Média: {mean_score:.3f} | Std: {std_score:.3f}")
        print("\n========== Grid Search Pendulum Finalizado ==========")
        return results