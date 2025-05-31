import yaml
import itertools
import numpy as np
import os
import logging
from datetime import datetime
from training.train_utils import (
    setup_logger,
    load_config,
    create_agent,
    run_training,
    run_training_multiple_seeds,
    analyze_training,
    save_grid_search_results,
    save_learning_curve_plot
)

OUTPUT_ANALYSIS_DIR = "output/analysis"

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

    print("\n========== Iniciando Grid Search ==========")
    for idx, (alpha, gamma, epsilon) in enumerate(param_grid, 1):
        for policy in policies:
            params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "policy": policy}
            msg = f"[{idx}/{len(param_grid)}] Parâmetros: {params}"
            print(msg)
            logging.info(msg)

            seeds = [config.get("seed", 42) + i for i in range(n_seeds)]
            mean_returns, agents, _ = run_training_multiple_seeds(
                agent_type=config["agent"]["type"],
                actions=[0, 1],
                params=params,
                episodes=config["training"]["episodes"],
                seeds=seeds,
                checkpoint_dir=f"output/checkpoints/{policy}/alpha{alpha}_gamma{gamma}_epsilon{epsilon}",
                checkpoint_interval=1000
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

def analyze_training(agent, save_dir=OUTPUT_ANALYSIS_DIR, policy_name=None):
    """
    Gera e salva análise gráfica do agente treinado:
    - Curva de aprendizagem (PNG e HTML)
    - Histograma dos retornos
    - Q-table e política aprendida (HTML)
    Funciona para qualquer agente que possua os atributos episode_returns e Q.
    O diretório de saída será nomeado conforme o argumento policy_name (ou outro nome passado).
    """
    import matplotlib.pyplot as plt
    from training.train_utils import save_learning_curve_plot

    # O nome da pasta será policy_name (ou qualquer nome passado)
    if policy_name:
        save_dir = os.path.join(save_dir, str(policy_name))
    os.makedirs(save_dir, exist_ok=True)

    # Curva de aprendizagem e histograma dos retornos
    if hasattr(agent, "episode_returns") and agent.episode_returns:
        returns = agent.episode_returns
        window = 200
        save_learning_curve_plot(
            returns,
            save_path=os.path.join(save_dir, "learning_curve.png"),
            window=window,
            title="Learning Curve"
        )

        import io, base64
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(returns) >= window:
            moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
            ax.plot(moving_avg, label=f"{window}-episode moving average")
        ax.plot(returns, alpha=0.3, label="Episode returns")
        ax.set_title("Learning Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        html_curve = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;">'
        with open(os.path.join(save_dir, "learning_curve.html"), "w", encoding="utf-8") as f:
            f.write(html_curve)

        plt.figure(figsize=(8, 4))
        plt.hist(returns, bins=50, color='skyblue', edgecolor='black')
        plt.title("Distribuição dos Retornos por Episódio")
        plt.xlabel("Retorno")
        plt.ylabel("Frequência")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "returns_histogram.png"))
        plt.close()

    # Q-table e política aprendida (HTML)
    if hasattr(agent, "Q"):
        from agents.utils import AgentVisualizer
        html_qtable = AgentVisualizer.plot_q_table(agent, title="Q-table do agente treinado")
        with open(os.path.join(save_dir, "qtable_heatmap.html"), "w", encoding="utf-8") as f:
            f.write(html_qtable)
        html_policy = AgentVisualizer.plot_policy(agent, title="Política aprendida")
        with open(os.path.join(save_dir, "policy_barplot.html"), "w", encoding="utf-8") as f:
            f.write(html_policy)

    print(f"\nRelatórios e gráficos salvos em: {os.path.abspath(save_dir)}")

if __name__ == "__main__":

    log_path = setup_logger()
    logging.info("Início do treino Blackjack RL")
    try:
        config = load_config("config/blackjack_config.yaml")
        policies = ["epsilon", "greedy", "softmax", "decay"]
        n_seeds = 5
        results = grid_search(config, policies=policies, n_seeds=n_seeds)
        save_grid_search_results(results, save_dir=OUTPUT_ANALYSIS_DIR)

        for params, mean_score, std_score, seeds_scores in results:
            policy_name = params.get("policy", "unknown")
            print(f"\nTreinando e analisando para policy: {policy_name}")

            mean_return, agent, results_counter = run_training(
                agent_type=config["agent"]["type"],
                actions=[0, 1],
                params=params,
                episodes=config["training"]["episodes"],
                seed=config.get("seed", 42),
                checkpoint_dir=f"output/checkpoints/{policy_name}",
                checkpoint_interval=1000,
                track_results=True
            )
            print(f"Retorno médio: {mean_return:.4f}")
            print(f"Vitórias: {results_counter['win']} | Derrotas: {results_counter['loss']} | Empates: {results_counter['draw']}")

            analyze_training(agent, save_dir=OUTPUT_ANALYSIS_DIR, policy_name=policy_name)

            with open(os.path.join(OUTPUT_ANALYSIS_DIR, policy_name, "metrics.txt"), "w", encoding="utf-8") as f:
                f.write(f"Retorno médio (primeiro seed): {mean_return:.4f}\n")
                f.write(f"Vitórias: {results_counter['win']}\n")
                f.write(f"Derrotas: {results_counter['loss']}\n")
                f.write(f"Empates: {results_counter['draw']}\n")
                f.write(f"Parâmetros: {params}\n")
                f.write(f"Retornos de todos os seeds: {seeds_scores}\n")
                f.write(f"Média dos seeds: {mean_score:.4f}\n")
                f.write(f"Desvio padrão dos seeds: {std_score:.4f}\n")

        logging.info("Treino finalizado com sucesso.")
    except Exception as e:
        print(f"\n[ERRO] Execução interrompida: {e}")
        logging.error(f"Execução interrompida: {e}", exc_info=True)
    print(f"\nLog salvo em: {log_path}")
    logging.info("Log salvo em: %s", log_path)
