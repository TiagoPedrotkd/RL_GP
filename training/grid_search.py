import itertools
import numpy as np
from training.train_utils import RLTrainingUtils

def grid_search(config, policies=None, n_seeds=3):
    """
    Executa uma grid search sobre os hiperparâmetros definidos no config.
    Para cada combinação, executa o treino com n_seeds diferentes e calcula média/desvio.
    """
    grid = config.get("grid", None)
    if grid:
        param_grid = list(itertools.product(
            grid.get("alpha", [0.1]),
            grid.get("gamma", [1.0]),
            grid.get("epsilon", [0.1])
        ))
    else:
        param_grid = [(config["agent"].get("alpha", 0.1),
                       config["agent"].get("gamma", 1.0),
                       config["agent"].get("epsilon", 0.1))]

    if policies is None:
        policies = [config["agent"].get("policy", "epsilon")]

    best_score = float('-inf')
    best_config = None
    best_agent = None
    results = []

    checkpoint_interval = 0

    print("\n========== Iniciando Grid Search ==========")
    for idx, (alpha, gamma, epsilon) in enumerate(param_grid, 1):
        for policy in policies:
            params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "policy": policy}
            msg = f"[{idx}/{len(param_grid)}] Parâmetros: {params}"
            print(msg)

            import logging
            logging.info(msg)

            seeds = [config.get("seed", 42) + i for i in range(n_seeds)]
            mean_returns, agents, _ = RLTrainingUtils.run_training_multiple_seeds(
                agent_type=config["agent"]["type"],
                actions=[0, 1],
                params=params,
                episodes=config["training"]["episodes"],
                seeds=seeds,
                checkpoint_dir=f"output/checkpoints/{policy}/alpha{alpha}_gamma{gamma}_epsilon{epsilon}",
                checkpoint_interval=checkpoint_interval
            )
            mean_score = float(np.mean(mean_returns))
            std_score = float(np.std(mean_returns))
            results.append((params, mean_score, std_score, mean_returns))
            print(f"  → Média: {mean_score:.4f} | Std: {std_score:.4f} | Seeds: {mean_returns}")
            logging.info(f"Média: {mean_score:.4f} | Std: {std_score:.4f} | Seeds: {mean_returns}")

            if mean_score > best_score:
                best_score = mean_score
                best_config = params
                best_agent = agents[np.argmax(mean_returns)]
                best_path = "output/best_qtable.npy"
                best_agent.save(best_path)
                print(f"  * Novo melhor agente salvo em: {best_path}")
                logging.info(f"Novo melhor agente salvo em: {best_path}")

    print("\n========== Grid Search Finalizado ==========")
    print(f"Melhor configuração: {best_config}")
    print(f"Melhor score: {best_score:.4f}")
    logging.info("\nMelhor configuração encontrada:")
    logging.info(f"{best_config}")
    logging.info(f"Score: {best_score:.4f}")

    return results
