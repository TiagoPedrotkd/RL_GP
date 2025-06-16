import logging
from training.train_utils import RLTrainingUtils
from training.grid_search import PendulumGridSearch
import os
import traceback
import yaml

OUTPUT_ANALYSIS_DIR = "output/analysis"
LOG_SAVED_MSG = "Log salvo em: %s"

QLEARNING_LOG_FILENAME = "train_pendulum_qlearning.log"
SARSA_LOG_FILENAME = "train_pendulum_sarsa.log"

def load_yaml_grid_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    qlearning_log = os.path.join("logs", QLEARNING_LOG_FILENAME)
    sarsa_log = os.path.join("logs", SARSA_LOG_FILENAME)

    qlearning_config = load_yaml_grid_config("config/pendulum_qlearning_config.yaml")
    sarsa_config = load_yaml_grid_config("config/pendulum_sarsa_config.yaml")

    env_config = {
        "make_env": lambda seed=None: __import__("environments.pendulum_env", fromlist=["make_pendulum_env"]).make_pendulum_env(seed=seed),
        "angle_buckets": 25,
        "vel_buckets": 25,
        "action_low": -2.0,
        "action_high": 2.0,
        "n_actions": 5
    }

    RLTrainingUtils.setup_logger(log_name=QLEARNING_LOG_FILENAME)
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(QLEARNING_LOG_FILENAME):
            logging.getLogger().removeHandler(handler)
    logging.info("Início do treino Pendulum RL - QLearning")
    try:
        param_grid = qlearning_config.get("grid", {})
        grid = PendulumGridSearch(env_config, param_grid, n_seeds=3, method="qlearning", verbose=True)
        results = grid.run()
        RLTrainingUtils.save_grid_search_results(results, save_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "pendulum_qlearning"))

        for params, mean_score, std_score, seeds_scores in results:
            policy = params.get("policy", "epsilon")
            policy_dir = os.path.join(OUTPUT_ANALYSIS_DIR, "pendulum_qlearning", policy)
            os.makedirs(policy_dir, exist_ok=True)

            env = env_config["make_env"](seed=42)
            from agents.pendulum_rl import PendulumRL
            rl = PendulumRL(
                angle_buckets=env_config.get("angle_buckets", 25),
                vel_buckets=env_config.get("vel_buckets", 25),
                action_low=env_config.get("action_low", -2.0),
                action_high=env_config.get("action_high", 2.0),
                n_actions=env_config.get("n_actions", 5)
            )

            agent, metrics = rl.train(
                env,
                method="qlearning",
                num_iterations=params.get("episodes", qlearning_config.get("training", {}).get("episodes", 1000)),
                alpha=params.get("alpha", 0.1),
                gamma=params.get("gamma", 0.99),
                epsilon=params.get("epsilon", 0.1),
                policy=policy,
                verbose=False
            )
            class DummyAgent:
                pass
            dummy = DummyAgent()
            dummy.Q = agent
            dummy.episode_returns = metrics["Return"]
            RLTrainingUtils.analyze_training(dummy, save_dir=policy_dir, policy_name=policy)

        logging.info("Treino QLearning Pendulum finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (Pendulum QLearning): {e}")
        traceback.print_exc()
    print(f"\nLog QLearning salvo em: {qlearning_log}")
    logging.info(LOG_SAVED_MSG, qlearning_log)

    RLTrainingUtils.setup_logger(log_name=SARSA_LOG_FILENAME)
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(SARSA_LOG_FILENAME):
            logging.getLogger().removeHandler(handler)
    logging.info("Início do treino Pendulum RL - SARSA")
    try:
        param_grid = sarsa_config.get("grid", {})
        grid = PendulumGridSearch(env_config, param_grid, n_seeds=3, method="sarsa", verbose=True)
        results = grid.run()
        RLTrainingUtils.save_grid_search_results(results, save_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "pendulum_sarsa"))
        for params, mean_score, std_score, seeds_scores in results:
            policy = params.get("policy", "epsilon")
            policy_dir = os.path.join(OUTPUT_ANALYSIS_DIR, "pendulum_sarsa", policy)
            os.makedirs(policy_dir, exist_ok=True)
            env = env_config["make_env"](seed=42)
            from agents.pendulum_rl import PendulumRL
            rl = PendulumRL(
                angle_buckets=env_config.get("angle_buckets", 25),
                vel_buckets=env_config.get("vel_buckets", 25),
                action_low=env_config.get("action_low", -2.0),
                action_high=env_config.get("action_high", 2.0),
                n_actions=env_config.get("n_actions", 5)
            )

            agent, metrics = rl.train(
                env,
                method="sarsa",
                num_iterations=params.get("episodes", sarsa_config.get("training", {}).get("episodes", 1000)),
                alpha=params.get("alpha", 0.1),
                gamma=params.get("gamma", 0.99),
                epsilon=params.get("epsilon", 0.1),
                policy=policy,
                verbose=False
            )
            class DummyAgent:
                pass
            dummy = DummyAgent()
            dummy.Q = agent
            dummy.episode_returns = metrics["Return"]
            RLTrainingUtils.analyze_training(dummy, save_dir=policy_dir, policy_name=policy)
        logging.info("Treino SARSA Pendulum finalizado com sucesso.")
    except Exception as e:
        import traceback
        print(f"\n[ERRO] Execução interrompida (Pendulum SARSA): {e}")
        traceback.print_exc()
    print(f"\nLog SARSA salvo em: {sarsa_log}")
    logging.info(LOG_SAVED_MSG, sarsa_log)

if __name__ == "__main__":
    main()
