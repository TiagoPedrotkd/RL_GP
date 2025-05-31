import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Optional

from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.monte_carlo import MonteCarloAgent
from environments.blackjack_env import make_blackjack_env

class RLTrainingUtils:
    LEARNING_CURVE_TITLE = "Learning Curve"

    @staticmethod
    def setup_logger(log_dir: str = "logs", log_name: str = "train_blackjack.log") -> str:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_name)
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        logging.info("Logger initialized.")
        return log_path

    @staticmethod
    def load_config(path: str = "config/blackjack_config.yaml") -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def create_agent(agent_type: str, actions: List[int], params: Dict[str, Any], seed: int) -> Any:
        agent_type_lower = agent_type.lower()
        policy = params.get("policy", None)
        if agent_type_lower == "qlearningagent":
            return QLearningAgent(
                actions,
                alpha=params.get("alpha", 0.1),
                gamma=params.get("gamma", 1.0),
                epsilon=params.get("epsilon", 0.1),
                policy=policy,
                seed=seed
            )
        elif agent_type_lower == "sarsaagent":
            return SARSAAgent(
                actions,
                alpha=params.get("alpha", 0.1),
                gamma=params.get("gamma", 1.0),
                epsilon=params.get("epsilon", 0.1),
                policy=policy,
                seed=seed
            )
        elif agent_type_lower == "montecarloagent":
            return MonteCarloAgent(
                actions,
                epsilon=params.get("epsilon", 0.1),
                gamma=params.get("gamma", 1.0),
                policy=policy,
                seed=seed
            )
        else:
            raise ValueError(f"Agente não suportado: {agent_type}")

    @staticmethod
    def run_monte_carlo_episode(env, agent, track_results: bool = False, results_counter: Optional[Dict[str, int]] = None) -> float:
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        done = False
        buffer = []
        reward = 0
        while not done:
            action = agent.policy(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            buffer.append((state, action, reward))
            state = next_state
        agent.update(buffer)
        if track_results and results_counter is not None:
            if reward > 0:
                results_counter["win"] += 1
            elif reward < 0:
                results_counter["loss"] += 1
            else:
                results_counter["draw"] += 1
        return sum([x[2] for x in buffer])

    @staticmethod
    def run_td_episode(env, agent, agent_type_lower: str) -> float:
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        done = False
        action = agent.policy(state)
        ep_return = 0
        while not done:
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            ep_return += reward
            next_action = agent.policy(next_state)
            if agent_type_lower == "sarsaagent":
                if done:
                    agent.update(state, action, reward, next_state, 0)
                else:
                    agent.update(state, action, reward, next_state, next_action)
            else:
                agent.update(state, action, reward, next_state)
            state, action = next_state, next_action

        if hasattr(agent, "episode_returns") and hasattr(env, "reward_range"):
            # Optionally, you could add logic here if needed in the future
            pass
        return ep_return

    @staticmethod
    def _run_training_episode(env, agent, agent_type_lower: str, track_results: bool, results_counter: Optional[Dict[str, int]]) -> float:
        if agent_type_lower == "montecarloagent":
            return RLTrainingUtils.run_monte_carlo_episode(
                env, agent,
                track_results=track_results,
                results_counter=results_counter if track_results else None
            )
        else:
            return RLTrainingUtils.run_td_episode(env, agent, agent_type_lower)

    @staticmethod
    def run_training(
        agent_type: str,
        actions: List[int],
        params: Dict[str, Any],
        episodes: int = 10000,
        seed: int = 42,
        checkpoint_dir: str = "../results/checkpoints",
        checkpoint_interval: int = 5000,
        track_results: bool = False
    ) -> Tuple[float, Any, Optional[Dict[str, int]]]:
        env = make_blackjack_env(seed=seed)
        agent = RLTrainingUtils.create_agent(agent_type, actions, params, seed)
        agent_type_lower = agent_type.lower()
        returns = []
        results_counter = {"win": 0, "loss": 0, "draw": 0}
        os.makedirs(checkpoint_dir, exist_ok=True)

        for ep in range(1, episodes + 1):
            try:
                ep_return = RLTrainingUtils._run_training_episode(
                    env, agent, agent_type_lower, track_results, results_counter
                )
                returns.append(ep_return)
                if ep % 1000 == 0:
                    logging.info(f"Episode {ep}/{episodes} | Mean Return (last 1000): {np.mean(returns[-1000:]):.3f}")
                    if track_results:
                        logging.info(f"Vitórias: {results_counter['win']} | Derrotas: {results_counter['loss']} | Empates: {results_counter['draw']}")
                if checkpoint_interval and ep % checkpoint_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"{agent_type}_ep{ep}.npy")
                    agent.save(checkpoint_path)
                    logging.info(f"Checkpoint salvo: {checkpoint_path}")
            except Exception as e:
                logging.error(f"Erro no episódio {ep}: {e}", exc_info=True)
                fail_path = os.path.join(checkpoint_dir, f"{agent_type}_FAILED_ep{ep}.npy")
                agent.save(fail_path)
                logging.info(f"Estado salvo em caso de erro: {fail_path}")
                break

        agent.episode_returns = returns

        if track_results:
            return np.mean(returns), agent, results_counter
        return np.mean(returns), agent

    @staticmethod
    def run_training_multiple_seeds(
        agent_type: str,
        actions: List[int],
        params: Dict[str, Any],
        episodes: int,
        seeds: List[int],
        checkpoint_dir: str,
        checkpoint_interval: int = 5000
    ) -> Tuple[List[float], List[Any], List[Dict[str, int]]]:
        mean_returns = []
        agents = []
        results_counters = []
        for seed in seeds:
            mean_return, agent, results_counter = RLTrainingUtils.run_training(
                agent_type=agent_type,
                actions=actions,
                params=params,
                episodes=episodes,
                seed=seed,
                checkpoint_dir=os.path.join(checkpoint_dir, f"seed_{seed}"),
                checkpoint_interval=checkpoint_interval,
                track_results=True
            )
            mean_returns.append(mean_return)
            agents.append(agent)
            results_counters.append(results_counter)
        return mean_returns, agents, results_counters

    @staticmethod
    def analyze_training(
        agent: Any,
        episode_returns: Optional[List[float]] = None,
        save_dir: str = "output/analysis",
        window: int = 200,
        policy_name: Optional[str] = None
    ) -> None:
        import io, base64
        if policy_name:
            save_dir = os.path.join(save_dir, str(policy_name))
        os.makedirs(save_dir, exist_ok=True)

        if episode_returns is None and hasattr(agent, "episode_returns"):
            episode_returns = agent.episode_returns
        if episode_returns:
            fig, ax = plt.subplots(figsize=(10, 5))
            if len(episode_returns) >= window:
                moving_avg = np.convolve(episode_returns, np.ones(window) / window, mode='valid')
                ax.plot(moving_avg, label=f"{window}-episode moving average")
            ax.plot(episode_returns, alpha=0.3, label="Episode returns")
            ax.set_title(RLTrainingUtils.LEARNING_CURVE_TITLE)
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
            plt.hist(episode_returns, bins=50, color='skyblue', edgecolor='black')
            plt.title("Distribuição dos Retornos por Episódio")
            plt.xlabel("Retorno")
            plt.ylabel("Frequência")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "returns_histogram.png"))
            plt.close()

        if hasattr(agent, "Q"):
            from agents.utils import AgentVisualizer
            html_qtable = AgentVisualizer.plot_q_table(agent, title="Q-table do agente treinado")
            with open(os.path.join(save_dir, "qtable_heatmap.html"), "w", encoding="utf-8") as f:
                f.write(html_qtable)
            html_policy = AgentVisualizer.plot_policy(agent, title="Política aprendida")
            with open(os.path.join(save_dir, "policy_barplot.html"), "w", encoding="utf-8") as f:
                f.write(html_policy)

        print(f"\nRelatórios e gráficos salvos em: {os.path.abspath(save_dir)}")

    @staticmethod
    def save_grid_search_results(
        results: List[Tuple[Dict[str, Any], float, float, List[float]]],
        save_dir: str = "output/analysis"
    ) -> None:
        import pandas as pd

        os.makedirs(save_dir, exist_ok=True)
        if not results:
            logging.warning("Nenhum resultado de grid search para salvar.")
            return

        df = pd.DataFrame([
            {
                "alpha": p.get("alpha"),
                "gamma": p.get("gamma"),
                "epsilon": p.get("epsilon"),
                "policy": p.get("policy"),
                "mean_score": mean,
                "std_score": std,
                "seeds": str(seeds)
            }
            for p, mean, std, seeds in results if p is not None
        ])
        df_sorted = df.sort_values("mean_score", ascending=False)

        df_sorted.to_html(os.path.join(save_dir, "grid_search_results.html"), index=False)
        df_sorted.to_csv(os.path.join(save_dir, "grid_search_results.csv"), index=False)

        plt.figure(figsize=(12, 6))
        plt.bar(
            range(len(df_sorted)),
            df_sorted["mean_score"],
            yerr=df_sorted["std_score"],
            color="royalblue",
            capsize=4
        )
        plt.xticks(
            range(len(df_sorted)),
            [f"{row.policy}\nα={row.alpha}, γ={row.gamma}, ε={row.epsilon}" for row in df_sorted.itertuples()],
            rotation=90
        )
        plt.ylabel("Score (média)")
        plt.title("Comparação dos Scores do Grid Search (média ± std)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "grid_search_scores.png"))
        plt.close()

    @staticmethod
    def save_learning_curve_plot(
        returns: List[float],
        save_path: str,
        window: int = 200,
        title: str = "Learning Curve"
    ) -> None:
        plt.figure(figsize=(10, 5))
        if len(returns) >= window:
            moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
            plt.plot(moving_avg, label=f"{window}-episode moving average")
        plt.plot(returns, alpha=0.3, label="Episode returns")
        plt.title(title if title else RLTrainingUtils.LEARNING_CURVE_TITLE)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def save_policy_metrics_html(
        policy_name: str,
        mean_return: float,
        results_counter: dict,
        params: dict,
        seeds_scores: list,
        mean_score: float,
        std_score: float,
        save_dir: str
    ):
        os.makedirs(save_dir, exist_ok=True)
        html_metrics = f"""
            <html>
                <head>
                    <title>Métricas - {policy_name}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        table {{ border-collapse: collapse; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ccc; padding: 8px 16px; }}
                        th {{ background: #f0f0f0; }}
                    </style>
                </head>
                <body>
                    <h2>Métricas - {policy_name}</h2>
                    <table>
                        <tr><th>Retorno médio (primeiro seed)</th><td>{mean_return:.4f}</td></tr>
                        <tr><th>Vitórias</th><td>{results_counter['win']}</td></tr>
                        <tr><th>Derrotas</th><td>{results_counter['loss']}</td></tr>
                        <tr><th>Empates</th><td>{results_counter['draw']}</td></tr>
                        <tr><th>Parâmetros</th><td><pre>{params}</pre></td></tr>
                        <tr><th>Retornos de todos os seeds</th><td>{seeds_scores}</td></tr>
                        <tr><th>Média dos seeds</th><td>{mean_score:.4f}</td></tr>
                        <tr><th>Desvio padrão dos seeds</th><td>{std_score:.4f}</td></tr>
                    </table>
                </body>
            </html>
        """
        with open(os.path.join(save_dir, "metrics.html"), "w", encoding="utf-8") as f:
            f.write(html_metrics)

    @staticmethod
    def run_experiment(config_path, agent_type, policies, output_dir, label=None):
        from training.grid_search import grid_search

        config = RLTrainingUtils.load_config(config_path)
        config["agent"]["type"] = agent_type
        n_seeds = 5
        results = grid_search(config, policies=policies, n_seeds=n_seeds)
        RLTrainingUtils.save_grid_search_results(results, save_dir=output_dir)

        for params, mean_score, std_score, seeds_scores in results:
            policy_name = params.get("policy", "unknown")
            policy_dir = os.path.join(output_dir, f"{label}_{policy_name}") if label else os.path.join(output_dir, policy_name)
            
            checkpoint_dir = os.path.join("output", "checkpoints", f"{label}_{policy_name}") if label else os.path.join("output", "checkpoints", policy_name)
            print(f"\nTreinando e analisando para policy: {policy_name}")

            mean_return, agent, results_counter = RLTrainingUtils.run_training(
                agent_type=agent_type,
                actions=[0, 1],
                params=params,
                episodes=config["training"]["episodes"],
                seed=config.get("seed", 42),
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=0,
                track_results=True
            )
            print(f"Retorno médio: {mean_return:.4f}")
            print(f"Vitórias: {results_counter['win']} | Derrotas: {results_counter['loss']} | Empates: {results_counter['draw']}")

            RLTrainingUtils.analyze_training(agent, save_dir=policy_dir, policy_name=None)

            with open(os.path.join(policy_dir, "metrics.txt"), "w", encoding="utf-8") as f:
                f.write(f"Retorno médio (primeiro seed): {mean_return:.4f}\n")
                f.write(f"Vitórias: {results_counter['win']}\n")
                f.write(f"Derrotas: {results_counter['loss']}\n")
                f.write(f"Empates: {results_counter['draw']}\n")
                f.write(f"Parâmetros: {params}\n")
                f.write(f"Retornos de todos os seeds: {seeds_scores}\n")
                f.write(f"Média dos seeds: {mean_score:.4f}\n")
                f.write(f"Desvio padrão dos seeds: {std_score:.4f}\n")

            RLTrainingUtils.save_policy_metrics_html(
                policy_name=policy_name,
                mean_return=mean_return,
                results_counter=results_counter,
                params=params,
                seeds_scores=seeds_scores,
                mean_score=mean_score,
                std_score=std_score,
                save_dir=policy_dir
            )
