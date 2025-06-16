import logging
from src.training.train_utils import RLTrainingUtils
from src.training.grid_search import BlackjackGridSearch
import os
import traceback
import numpy as np

OUTPUT_ANALYSIS_DIR = "output/analysis"
LOG_SAVED_MSG = "Log salvo em: %s"

MONTECARLO_LOG_FILENAME = "train_blackjack_montecarlo.log"
SARSA_LOG_FILENAME = "train_blackjack_sarsa.log"
QLEARNING_LOG_FILENAME = "train_blackjack_qlearning.log"

def analyze_and_update_learning_curves(agent, policy, policy_dir, window, learning_curves):
    RLTrainingUtils.analyze_training(agent, save_dir=policy_dir, policy_name=policy)
    if hasattr(agent, "episode_returns") and agent.episode_returns:
        returns = agent.episode_returns
        if len(returns) >= window:
            moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
        else:
            moving_avg = np.array(returns)
        learning_curves[policy] = moving_avg

def run_rl_training(
    log_filename,
    config_path,
    output_subdir,
    log_saved_msg,
    log_path,
    n_seeds=3
):
    RLTrainingUtils.setup_logger(log_name=log_filename)
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(log_filename):
            logging.getLogger().removeHandler(handler)
    logging.info(f"Início do treino Blackjack RL - {output_subdir.capitalize()}")
    try:
        config = RLTrainingUtils.load_config(config_path)
        grid = BlackjackGridSearch(config, n_seeds=n_seeds, verbose=True)
        results = grid.run()
        analysis_dir = os.path.join(OUTPUT_ANALYSIS_DIR, output_subdir)
        RLTrainingUtils.save_grid_search_results(results, save_dir=analysis_dir)
        print(f"\n========== Análise Final {output_subdir.capitalize()} ==========")
        learning_curves = {}
        window = 200
        for params, mean_score, std_score, mean_returns in results:
            print(f"Parâmetros: {params}")
            print(f"  Média das seeds: {mean_score:.4f} | Std: {std_score:.4f}")
            print(f"  Retornos das seeds: {mean_returns}")

            policy = params.get("policy", "unknown")
            policy_dir = os.path.join(analysis_dir, policy)
            os.makedirs(policy_dir, exist_ok=True)

            agent_type = config["agent"]["type"]
            actions = [0, 1]
            episodes = params.get("episodes", config["training"]["episodes"])
            _, agent, _ = RLTrainingUtils.run_training(
                agent_type=agent_type,
                actions=actions,
                params=params,
                config={
                    "training": {"episodes": episodes},
                    "env_name": config.get("env_name", "Blackjack-v1"),
                    "seed": config.get("seed", 42)
                }
            )
            analyze_and_update_learning_curves(agent, policy, policy_dir, window, learning_curves)

        if learning_curves:
            RLTrainingUtils.plot_learning_curves_comparison(
                learning_curves,
                save_path=os.path.join(analysis_dir, "learning_curves_comparison.png"),
                window=window
            )

        random_policy_dir = os.path.join(analysis_dir, "random")
        os.makedirs(random_policy_dir, exist_ok=True)
        random_config = {
            "training": {
                "episodes": config["training"]["episodes"],
                "checkpoint_dir": os.path.join("output", "checkpoints", f"{output_subdir}_random"),
                "checkpoint_interval": 0,
                "track_results": True,
                "max_steps_per_episode": config["training"].get("max_steps_per_episode", 100),
                "early_stopping": config["training"].get("early_stopping", False),
                "early_stopping_patience": config["training"].get("early_stopping_patience", 1000),
                "early_stopping_delta": config["training"].get("early_stopping_delta", 0.01),
                "save_best": config["training"].get("save_best", True),
                "verbose": config["training"].get("verbose", True)
            },
            "env_name": config.get("env_name", "Blackjack-v1"),
            "seed": config.get("seed", 42)
        }
        _, random_agent, _ = RLTrainingUtils.run_training(
            agent_type="RandomAgent",
            actions=[0, 1],
            params={},
            config=random_config
        )
        analyze_and_update_learning_curves(random_agent, "random", random_policy_dir, window, learning_curves)

        if "random" in learning_curves:
            RLTrainingUtils.plot_learning_curves_comparison(
                learning_curves,
                save_path=os.path.join(analysis_dir, "learning_curves_comparison.png"),
                window=window
            )

        logging.info(f"Treino {output_subdir.capitalize()} finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida ({output_subdir.capitalize()}): {e}")
        traceback.print_exc()
    print(f"\nLog {output_subdir.capitalize()} salvo em: {log_path}")
    logging.info(log_saved_msg, log_path)


def main():
    montecarlo_log = os.path.join("logs", MONTECARLO_LOG_FILENAME)
    sarsa_log = os.path.join("logs", SARSA_LOG_FILENAME)
    qlearning_log = os.path.join("logs", QLEARNING_LOG_FILENAME)

    n_seeds = 3

    run_rl_training(
        log_filename=MONTECARLO_LOG_FILENAME,
        config_path="config/blackjack_montecarlo_config.yaml",
        output_subdir="montecarlo",
        log_saved_msg=LOG_SAVED_MSG,
        log_path=montecarlo_log,
        n_seeds=n_seeds
    )

    run_rl_training(
        log_filename=SARSA_LOG_FILENAME,
        config_path="config/blackjack_sarsa_config.yaml",
        output_subdir="sarsa",
        log_saved_msg=LOG_SAVED_MSG,
        log_path=sarsa_log,
        n_seeds=n_seeds
    )

    run_rl_training(
        log_filename=QLEARNING_LOG_FILENAME,
        config_path="config/blackjack_qlearning_config.yaml",
        output_subdir="qlearning",
        log_saved_msg=LOG_SAVED_MSG,
        log_path=qlearning_log,
        n_seeds=n_seeds
    )

if __name__ == "__main__":
    main()