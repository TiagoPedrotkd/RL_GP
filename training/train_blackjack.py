import logging
from training.train_utils import RLTrainingUtils
from training.grid_search import BlackjackGridSearch
import os
import traceback

OUTPUT_ANALYSIS_DIR = "output/analysis"
LOG_SAVED_MSG = "Log salvo em: %s"

MONTECARLO_LOG_FILENAME = "train_blackjack_montecarlo.log"
SARSA_LOG_FILENAME = "train_blackjack_sarsa.log"
QLEARNING_LOG_FILENAME = "train_blackjack_qlearning.log"

def main():
    montecarlo_log = os.path.join("logs", MONTECARLO_LOG_FILENAME)
    sarsa_log = os.path.join("logs", SARSA_LOG_FILENAME)
    qlearning_log = os.path.join("logs", QLEARNING_LOG_FILENAME)

    n_seeds = 3

    # Monte Carlo
    RLTrainingUtils.setup_logger(log_name=MONTECARLO_LOG_FILENAME)
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(MONTECARLO_LOG_FILENAME):
            logging.getLogger().removeHandler(handler)
    logging.info("Início do treino Blackjack RL - MonteCarlo")
    try:
        config = RLTrainingUtils.load_config("config/blackjack_montecarlo_config.yaml")
        grid = BlackjackGridSearch(config, n_seeds=n_seeds, verbose=True)
        results = grid.run()
        RLTrainingUtils.save_grid_search_results(results, save_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "montecarlo"))
        print("\n========== Análise Final MonteCarlo ==========")
        for params, mean_score, std_score, mean_returns in results:
            print(f"Parâmetros: {params}")
            print(f"  Média das seeds: {mean_score:.4f} | Std: {std_score:.4f}")
            print(f"  Retornos das seeds: {mean_returns}")
        logging.info("Treino MonteCarlo finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (MonteCarlo): {e}")
        traceback.print_exc()
    print(f"\nLog MonteCarlo salvo em: {montecarlo_log}")
    logging.info(LOG_SAVED_MSG, montecarlo_log)

    # SARSA
    RLTrainingUtils.setup_logger(log_name=SARSA_LOG_FILENAME)
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(SARSA_LOG_FILENAME):
            logging.getLogger().removeHandler(handler)
    logging.info("Início do treino Blackjack RL - SARSA")
    try:
        config = RLTrainingUtils.load_config("config/blackjack_sarsa_config.yaml")
        grid = BlackjackGridSearch(config, n_seeds=n_seeds, verbose=True)
        results = grid.run()
        RLTrainingUtils.save_grid_search_results(results, save_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "sarsa"))
        print("\n========== Análise Final SARSA ==========")
        for params, mean_score, std_score, mean_returns in results:
            print(f"Parâmetros: {params}")
            print(f"  Média das seeds: {mean_score:.4f} | Std: {std_score:.4f}")
            print(f"  Retornos das seeds: {mean_returns}")
        logging.info("Treino SARSA finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (SARSA): {e}")
        traceback.print_exc()
    print(f"\nLog SARSA salvo em: {sarsa_log}")
    logging.info(LOG_SAVED_MSG, sarsa_log)

    # Q-Learning
    RLTrainingUtils.setup_logger(log_name=QLEARNING_LOG_FILENAME)
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler) and not handler.baseFilename.endswith(QLEARNING_LOG_FILENAME):
            logging.getLogger().removeHandler(handler)
    logging.info("Início do treino Blackjack RL - QLearning")
    try:
        config = RLTrainingUtils.load_config("config/blackjack_qlearning_config.yaml")
        grid = BlackjackGridSearch(config, n_seeds=n_seeds, verbose=True)
        results = grid.run()
        RLTrainingUtils.save_grid_search_results(results, save_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "qlearning"))
        print("\n========== Análise Final QLearning ==========")
        for params, mean_score, std_score, mean_returns in results:
            print(f"Parâmetros: {params}")
            print(f"  Média das seeds: {mean_score:.4f} | Std: {std_score:.4f}")
            print(f"  Retornos das seeds: {mean_returns}")
        logging.info("Treino QLearning finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida (QLearning): {e}")
        traceback.print_exc()
    print(f"\nLog QLearning salvo em: {qlearning_log}")
    logging.info(LOG_SAVED_MSG, qlearning_log)

if __name__ == "__main__":
    main()