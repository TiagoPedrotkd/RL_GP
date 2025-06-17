import logging
from src.training.train_utils import RLTrainingUtils
from src.training.grid_search import PendulumGridSearch
import os
import traceback
import yaml
import numpy as np

OUTPUT_ANALYSIS_DIR = "output/analysis"
LOG_SAVED_MSG = "Log salvo em: %s"

QLEARNING_LOG_FILENAME = "train_pendulum_qlearning.log"
SARSA_LOG_FILENAME = "train_pendulum_sarsa.log"

def load_yaml_grid_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_training(
    method_name,
    log_filename,
    config,
    env_config,
    output_analysis_dir,
    method_key
):
    log_path = os.path.join("logs", log_filename)
    RLTrainingUtils.setup_logger(log_name=log_filename)
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(log_filename):
            logging.getLogger().removeHandler(handler)
    logging.info(f"Início do treino Pendulum RL - {method_name}")
    try:
        param_grid = config.get("grid", {})
        grid = PendulumGridSearch(env_config, param_grid, n_seeds=3, method=method_key, verbose=True)
        results = grid.run()
        analysis_dir = os.path.join(output_analysis_dir, f"pendulum_{method_key}")
        RLTrainingUtils.save_grid_search_results(results, save_dir=analysis_dir)

        learning_curves = {}
        window = 200

        for params, mean_score, std_score, seeds_scores in results:
            policy = params.get("policy", "epsilon")
            policy_dir = os.path.join(analysis_dir, policy)
            os.makedirs(policy_dir, exist_ok=True)

            env = env_config["make_env"](seed=42)
            from src.agents.pendulum_rl import PendulumRL
            rl = PendulumRL(
                angle_buckets=env_config.get("angle_buckets", 25),
                vel_buckets=env_config.get("vel_buckets", 25),
                action_low=env_config.get("action_low", -2.0),
                action_high=env_config.get("action_high", 2.0),
                n_actions=env_config.get("n_actions", 5)
            )

            agent, metrics = rl.train(
                env,
                method=method_key,
                num_iterations=params.get("episodes", config.get("training", {}).get("episodes", 1000)),
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

            returns = metrics["Return"]
            if len(returns) >= window:
                moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
            else:
                moving_avg = np.array(returns)
            learning_curves[policy] = moving_avg
            RLTrainingUtils.save_learning_curve_plot(
                returns,
                save_path=os.path.join(policy_dir, "learning_curve.png"),
                window=window,
                title=f"Learning Curve - {policy.capitalize()}"
            )

        if learning_curves:
            RLTrainingUtils.plot_learning_curves_comparison(
                learning_curves,
                save_path=os.path.join(analysis_dir, "learning_curves_comparison.png"),
                window=window
            )

        logging.info(f"Treino {method_name} Pendulum finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (Pendulum {method_name}): {e}")
        traceback.print_exc()
    print(f"\nLog {method_name} salvo em: {log_path}")
    logging.info(LOG_SAVED_MSG, log_path)

def main():

    qlearning_config = load_yaml_grid_config("config/pendulum_qlearning_config.yaml")
    sarsa_config = load_yaml_grid_config("config/pendulum_sarsa_config.yaml")

    from src.environments.pendulum_env import make_pendulum_env

    env_config = {
        "make_env": lambda seed=None: make_pendulum_env(seed=seed),
        "angle_buckets": 25,
        "vel_buckets": 25,
        "action_low": -2.0,
        "action_high": 2.0,
        "n_actions": 5
    }

    run_training(
        method_name="QLearning",
        log_filename=QLEARNING_LOG_FILENAME,
        config=qlearning_config,
        env_config=env_config,
        output_analysis_dir=OUTPUT_ANALYSIS_DIR,
        method_key="qlearning"
    )

    run_training(
        method_name="SARSA",
        log_filename=SARSA_LOG_FILENAME,
        config=sarsa_config,
        env_config=env_config,
        output_analysis_dir=OUTPUT_ANALYSIS_DIR,
        method_key="sarsa"
    )

if __name__ == "__main__":
    main()
