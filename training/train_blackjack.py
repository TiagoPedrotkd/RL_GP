import logging
from training.train_utils import RLTrainingUtils
from training.grid_search import grid_search
import os

OUTPUT_ANALYSIS_DIR = "output/analysis"
LOG_SAVED_MSG = "Log salvo em: %s"

def main():
    montecarlo_log = os.path.join("logs", "train_blackjack_montecarlo.log")
    sarsa_log = os.path.join("logs", "train_blackjack_sarsa.log")
    qlearning_log = os.path.join("logs", "train_blackjack_qlearning.log")

    # Monte Carlo
    RLTrainingUtils.setup_logger(log_name="train_blackjack_montecarlo.log")
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and h.baseFilename.endswith("train_blackjack_montecarlo.log")]  # Ensure only this log
    logging.info("Início do treino Blackjack RL - MonteCarlo")
    try:
        RLTrainingUtils.run_experiment(
            config_path="config/blackjack_montecarlo_config.yaml",
            agent_type="MonteCarloAgent",
            policies=["epsilon", "greedy", "softmax", "decay"],
            output_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "montecarlo"),
            label="montecarlo"
        )
        logging.info("Treino MonteCarlo finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (MonteCarlo): {e}")
    print(f"\nLog MonteCarlo salvo em: {montecarlo_log}")
    logging.info(LOG_SAVED_MSG, montecarlo_log)

    # SARSA
    RLTrainingUtils.setup_logger(log_name="train_blackjack_sarsa.log")
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and h.baseFilename.endswith("train_blackjack_sarsa.log")]
    logging.info("Início do treino Blackjack RL - SARSA")
    try:
        RLTrainingUtils.run_experiment(
            config_path="config/blackjack_sarsa_config.yaml",
            agent_type="SARSAAgent",
            policies=["epsilon", "greedy", "softmax", "decay"],
            output_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "sarsa"),
            label="sarsa"
        )
        logging.info("Treino SARSA finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (SARSA): {e}")
    print(f"\nLog SARSA salvo em: {sarsa_log}")

    # Q-Learning
    RLTrainingUtils.setup_logger(log_name="train_blackjack_qlearning.log")
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler) and h.baseFilename.endswith("train_blackjack_qlearning.log")]
    logging.info("Início do treino Blackjack RL - QLearning")
    try:
        RLTrainingUtils.run_experiment(
            config_path="config/blackjack_qlearning_config.yaml",
            agent_type="QLearningAgent",
            policies=["epsilon", "greedy", "softmax", "decay"],
            output_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "qlearning"),
            label="qlearning"
        )
        logging.info("Treino QLearning finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (QLearning): {e}")
    print(f"\nLog QLearning salvo em: {qlearning_log}")
    logging.info(LOG_SAVED_MSG, qlearning_log)

if __name__ == "__main__":
    main()