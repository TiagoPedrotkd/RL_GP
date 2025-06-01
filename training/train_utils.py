import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.monte_carlo import MonteCarloAgent
from agents.random_agent import RandomAgent
from environments.blackjack_env import make_env

@dataclass
class ReportMetadata:
    mean_score: float
    std_score: float
    seeds_scores: List[float]
    config: Dict
    output_dir: str
    label: Optional[str]
    window: int
    learning_curves: Dict[str, List[float]]

class TrainingConfig:
    agent_type: str
    env_name: str
    seed: int
    episodes: int
    max_steps_per_episode: int
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_delta: float
    save_best: bool
    verbose: bool

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
        elif agent_type_lower == "randomagent":
            return RandomAgent(actions, seed=seed)
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
    def run_td_episode(env, agent, agent_type_lower: str, track_results: bool = False, results_counter: Optional[Dict[str, int]] = None) -> float:
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        done = False
        action = agent.policy(state)
        ep_return = 0
        last_reward = 0
        while not done:
            next_state, reward, done = RLTrainingUtils._step_env(env, action)
            ep_return += reward
            last_reward = reward
            next_action = agent.policy(next_state)
            RLTrainingUtils._update_td_agent(agent, agent_type_lower, state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action

        RLTrainingUtils._track_results(last_reward, track_results, results_counter)
        return ep_return
    
    @staticmethod
    def _step_env(env, action):
        """
        Compatibiliza o retorno do método env.step(action) para diferentes versões do Gym.
        Garante o retorno sempre como (next_state, reward, done)
        """
        step_result = env.step(action)
        if len(step_result) == 5:  # Gym >=0.26
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        elif len(step_result) == 4:  # Gym <=0.25
            next_state, reward, done, _ = step_result
        else:
            raise ValueError(f"Formato inesperado no retorno do ambiente: {step_result}")
        return next_state, reward, done

    @staticmethod
    def _update_td_agent(agent, agent_type_lower, state, action, reward, next_state, next_action, done):
        if agent_type_lower == "sarsaagent":
            if done:
                agent.update(state, action, reward, next_state, 0)
            else:
                agent.update(state, action, reward, next_state, next_action)
        else:
            agent.update(state, action, reward, next_state)

    @staticmethod
    def _run_training_episode(env, agent, agent_type_lower: str, track_results: bool, results_counter: Optional[Dict[str, int]] = None) -> float:
        if agent_type_lower == "montecarloagent":
            return RLTrainingUtils.run_monte_carlo_episode(
                env, agent,
                track_results=track_results,
                results_counter=results_counter if track_results else None
            )
        else:
            return RLTrainingUtils.run_td_episode(env, agent, agent_type_lower, track_results, results_counter)

    @staticmethod
    def set_global_seed(seed: int):
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _setup_training(agent_type, actions, params, seed, config):
        if config is None:
            config = {}
        checkpoint_dir = config.get("checkpoint_dir", "../results/checkpoints")
        checkpoint_interval = config.get("checkpoint_interval", 5000)
        track_results = config.get("track_results", False)
        env_name = config.get("env_name", "Blackjack-v1")
        max_steps_per_episode = config.get("max_steps_per_episode", 100)
        early_stopping = config.get("early_stopping", False)
        early_stopping_patience = config.get("early_stopping_patience", 1000)
        early_stopping_delta = config.get("early_stopping_delta", 0.01)
        save_best = config.get("save_best", True)
        verbose = config.get("verbose", True)
        RLTrainingUtils.set_global_seed(seed)
        env = make_env(env_name=env_name, seed=seed)
        agent = RLTrainingUtils.create_agent(agent_type, actions, params, seed)
        agent_type_lower = agent_type.lower()
        os.makedirs(checkpoint_dir, exist_ok=True)
        return (env, agent, agent_type_lower, checkpoint_dir, checkpoint_interval, track_results,
                max_steps_per_episode, early_stopping, early_stopping_patience, early_stopping_delta,
                save_best, verbose, env_name)
    
    @staticmethod
    def _run_episode(env, agent, agent_type_lower, max_steps_per_episode, track_results, results_counter):
        if max_steps_per_episode is not None and max_steps_per_episode > 0:
            return RLTrainingUtils._run_limited_steps_episode(
                env, agent, agent_type_lower, max_steps_per_episode, track_results, results_counter
            )
        else:
            return RLTrainingUtils._run_training_episode(
                env, agent, agent_type_lower, track_results, results_counter
            )

    @staticmethod
    def _run_limited_steps_episode(env, agent, agent_type_lower, max_steps, track_results, results_counter):
        # Special handling for MonteCarloAgent: use its own episode logic
        if agent_type_lower == "montecarloagent":
            return RLTrainingUtils.run_monte_carlo_episode(
                env, agent,
                track_results=track_results,
                results_counter=results_counter
            )
        # TD agents (Q-Learning, SARSA)
        steps = 0
        done = False
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        action = agent.policy(state)
        ep_return = 0
        reward = 0
        while not done and steps < max_steps:
            next_state, reward, done = RLTrainingUtils._step_env(env, action)
            ep_return += reward
            next_action = agent.policy(next_state)
            RLTrainingUtils._update_td_agent(agent, agent_type_lower, state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            steps += 1
        RLTrainingUtils._track_results(reward, track_results, results_counter)
        return ep_return

    @staticmethod
    def _update_agent(agent, agent_type_lower, state, action, reward, next_state, next_action, done):
        # Only call for TD agents; MonteCarloAgent handled separately
        RLTrainingUtils._update_td_agent(agent, agent_type_lower, state, action, reward, next_state, next_action, done)

    @staticmethod
    def _track_results(reward, track_results, results_counter):
        if track_results and results_counter is not None:
            if reward > 0:
                results_counter["win"] += 1
            elif reward < 0:
                results_counter["loss"] += 1
            else:
                results_counter["draw"] += 1
            
    @staticmethod
    def _maybe_checkpoint(agent, agent_type, ep, checkpoint_interval, checkpoint_dir, verbose):
        if checkpoint_interval and ep % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{agent_type}_ep{ep}.npy")
            agent.save(checkpoint_path)
            logging.info(f"Checkpoint salvo: {checkpoint_path}")

    @staticmethod
    def _early_stopping_logic(returns, ep, early_stopping, early_stopping_patience, early_stopping_delta,
                              best_mean, patience_counter, save_best, agent, agent_type, checkpoint_dir):
        stop = False
        if early_stopping and ep >= early_stopping_patience:
            recent_returns = returns[-early_stopping_patience:]
            mean_recent = np.mean(recent_returns)
            if best_mean is None or mean_recent > best_mean + early_stopping_delta:
                best_mean = mean_recent
                patience_counter = 0
                if save_best:
                    best_path = os.path.join(checkpoint_dir, f"{agent_type}_best.npy")
                    agent.save(best_path)
                    logging.info(f"Novo melhor agente salvo em: {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping ativado no episódio {ep}.")
                    stop = True
        return best_mean, patience_counter, stop
    
    @staticmethod
    def _log_progress(ep, episodes, returns, verbose, track_results, results_counter):
        if verbose and ep % 1000 == 0:
            logging.info(f"Episode {ep}/{episodes} | Mean Return (last 1000): {np.mean(returns[-1000:]):.3f}")
            if track_results:
                logging.info(f"Vitórias: {results_counter['win']} | Derrotas: {results_counter['loss']} | Empates: {results_counter['draw']}")

    @staticmethod
    def run_training(
        agent_type: str,
        actions: List[int],
        params: Dict[str, Any],
        episodes: int = 10000,
        seed: int = 42,
        checkpoint_dir: str = "../results/checkpoints",
        checkpoint_interval: int = 5000,
        track_results: bool = False,
        env_name: str = "Blackjack-v1",
        max_steps_per_episode: int = 100,
        early_stopping: bool = False,
        early_stopping_patience: int = 1000,
        early_stopping_delta: float = 0.01,
        save_best: bool = True,
        verbose: bool = True
    ) -> Tuple[float, Any, Optional[Dict[str, int]]]:
        (env, agent, agent_type_lower, _, _, _, _, _, _, _, _, _, _) = RLTrainingUtils._setup_training(
            agent_type, actions, params, seed,
            {
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_interval": checkpoint_interval,
                "track_results": track_results,
                "env_name": env_name,
                "max_steps_per_episode": max_steps_per_episode,
                "early_stopping": early_stopping,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_delta": early_stopping_delta,
                "save_best": save_best,
                "verbose": verbose
            }
        )

        returns = []
        results_counter = {"win": 0, "loss": 0, "draw": 0}
        best_mean = None
        patience_counter = 0

        for ep in range(1, episodes + 1):
            try:
                ep_return = RLTrainingUtils._run_episode(env, agent, agent_type_lower, max_steps_per_episode, track_results, results_counter)
                returns.append(ep_return)
                RLTrainingUtils._log_progress(ep, episodes, returns, verbose, track_results, results_counter)
                RLTrainingUtils._maybe_checkpoint(agent, agent_type, ep, checkpoint_interval, checkpoint_dir, verbose)
                best_mean, patience_counter, stop = RLTrainingUtils._early_stopping_logic(
                    returns, ep, early_stopping, early_stopping_patience, early_stopping_delta,
                    best_mean, patience_counter, save_best, agent, agent_type, checkpoint_dir
                )
                if stop:
                    break
            except Exception as e:
                logging.error(f"Erro no episódio {ep}: {e}", exc_info=True)
                fail_path = os.path.join(checkpoint_dir, f"{agent_type}_FAILED_ep{ep}.npy")
                agent.save(fail_path)
                logging.info(f"Estado salvo em caso de erro: {fail_path}")
                break

        agent.episode_returns = returns

        if track_results:
            return np.mean(returns), agent, results_counter
        else:
            return np.mean(returns), agent, None

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
            # Learning curve with moving average
            fig, ax = plt.subplots(figsize=(12, 6))
            if len(episode_returns) >= window:
                moving_avg = np.convolve(episode_returns, np.ones(window) / window, mode='valid')
                ax.plot(moving_avg, label=f"{window}-episode moving average", color="royalblue", linewidth=2)
            ax.plot(episode_returns, alpha=0.3, label="Episode returns", color="gray")
            ax.set_title("Learning Curve", fontsize=16)
            ax.set_xlabel("Episode", fontsize=14)
            ax.set_ylabel("Return", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            html_curve = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;">'
            with open(os.path.join(save_dir, "learning_curve.html"), "w", encoding="utf-8") as f:
                f.write(html_curve)

            # Histogram of returns
            plt.figure(figsize=(10, 5))
            plt.hist(episode_returns, bins=50, color='#4F81BD', edgecolor='black', alpha=0.85)
            plt.title("Distribution of Episode Returns", fontsize=16)
            plt.xlabel("Return", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "returns_histogram.png"))
            plt.close()

        if hasattr(agent, "Q"):
            from agents.utils import AgentVisualizer
            html_qtable = AgentVisualizer.plot_q_table(agent, title="Trained Agent Q-table")
            with open(os.path.join(save_dir, "qtable_heatmap.html"), "w", encoding="utf-8") as f:
                f.write(html_qtable)
            html_policy = AgentVisualizer.plot_policy(agent, title="Learned Policy")
            with open(os.path.join(save_dir, "policy_barplot.html"), "w", encoding="utf-8") as f:
                f.write(html_policy)

        print(f"\nReports and plots saved at: {os.path.abspath(save_dir)}")

    @staticmethod
    def save_grid_search_results(
        results: List[Tuple[Dict[str, Any], float, float, List[float]]],
        save_dir: str = "output/analysis"
    ) -> None:
        import pandas as pd

        os.makedirs(save_dir, exist_ok=True)
        if not results:
            logging.warning("No grid search results to save.")
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

        plt.figure(figsize=(14, 7))
        plt.bar(
            range(len(df_sorted)),
            df_sorted["mean_score"],
            yerr=df_sorted["std_score"],
            color="#4F81BD",
            capsize=4,
            edgecolor='black'
        )
        plt.xticks(
            range(len(df_sorted)),
            [f"{row.policy}\nα={row.alpha}, γ={row.gamma}, ε={row.epsilon}" for row in df_sorted.itertuples()],
            rotation=45,
            ha='right'
        )
        plt.ylabel("Mean Score", fontsize=14)
        plt.title("Grid Search Score Comparison (mean ± std)", fontsize=16)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(save_dir, "grid_search_scores.png"))
        plt.close()

    @staticmethod
    def save_learning_curve_plot(
        returns: List[float],
        save_path: str,
        window: int = 200,
        title: str = "Learning Curve"
    ) -> None:
        plt.figure(figsize=(12, 6))
        if len(returns) >= window:
            moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
            plt.plot(moving_avg, label=f"{window}-episode moving average", color="royalblue", linewidth=2)
        plt.plot(returns, alpha=0.3, label="Episode returns", color="gray")
        plt.title(title, fontsize=16)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Return", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_learning_curves_comparison(learning_curves: dict, save_path: str, window: int = 200):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7))
        # Sort policies for consistent color mapping
        sorted_policies = sorted(learning_curves.keys())
        colors = plt.cm.get_cmap('tab10', len(sorted_policies))
        for idx, policy in enumerate(sorted_policies):
            curve = learning_curves[policy]
            plt.plot(
                curve,
                label=policy.capitalize(),
                linewidth=2,
                color=colors(idx)
            )
        plt.title(f"Learning Curves Comparison ({window}-episode moving average)", fontsize=18)
        plt.xlabel("Episode", fontsize=15)
        plt.ylabel("Return (moving average)", fontsize=15)
        plt.legend(title="Policy", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    @staticmethod
    def run_experiment(config_path, agent_type, policies, output_dir, label=None):
        from training.grid_search import grid_search

        config = RLTrainingUtils.load_config(config_path)
        config["agent"]["type"] = agent_type
        env_name, seed, train_cfg = RLTrainingUtils._extract_experiment_config(config)
        episodes = train_cfg.get("episodes", 10000)
        max_steps_per_episode = train_cfg.get("max_steps_per_episode", 100)
        early_stopping = train_cfg.get("early_stopping", False)
        early_stopping_patience = train_cfg.get("early_stopping_patience", 1000)
        early_stopping_delta = train_cfg.get("early_stopping_delta", 0.01)
        save_best = train_cfg.get("save_best", True)
        verbose = train_cfg.get("verbose", True)
        n_seeds = 5
        results = grid_search(config, policies=policies, n_seeds=n_seeds)
        RLTrainingUtils.save_grid_search_results(results, save_dir=output_dir)

        learning_curves = {}
        window = 200

        for params, mean_score, std_score, seeds_scores in results:
            report_meta = ReportMetadata(
                mean_score=mean_score,
                std_score=std_score,
                seeds_scores=seeds_scores,
                config=config,
                output_dir=output_dir,
                label=label,
                window=window,
                learning_curves=learning_curves
            )

            training_cfg = TrainingConfig()
            training_cfg.agent_type = agent_type
            training_cfg.env_name = env_name
            training_cfg.seed = seed
            training_cfg.episodes = episodes
            training_cfg.max_steps_per_episode = max_steps_per_episode
            training_cfg.early_stopping = early_stopping
            training_cfg.early_stopping_patience = early_stopping_patience
            training_cfg.early_stopping_delta = early_stopping_delta
            training_cfg.save_best = save_best
            training_cfg.verbose = verbose

            RLTrainingUtils.train_and_report_policy(
                params=params,
                training_cfg=training_cfg,
                report_meta=report_meta
            )

        RLTrainingUtils.train_and_report_random(
            config=config,
            output_dir=output_dir,
            label=label,
            window=window,
            env_name=env_name,
            seed=seed,
            learning_curves=learning_curves
        )

        RLTrainingUtils.plot_learning_curves_comparison(
            learning_curves,
            save_path=os.path.join(output_dir, "learning_curves_comparison.png"),
            window=window
        )

    @staticmethod
    def _extract_experiment_config(config):
        env_name = config.get("env_name", "Blackjack-v1")
        seed = config.get("seed", 42)
        train_cfg = config.get("training", {})
        return env_name, seed, train_cfg

    @staticmethod
    def train_and_report_policy(params: Dict, training_cfg: TrainingConfig, report_meta: ReportMetadata):
        policy_name = params.get("policy", "unknown")
        policy_dir = os.path.join(report_meta.output_dir, f"{report_meta.label}_{policy_name}") if report_meta.label else os.path.join(report_meta.output_dir, policy_name)
        checkpoint_dir = os.path.join("output", "checkpoints", f"{report_meta.label}_{policy_name}") if report_meta.label else os.path.join("output", "checkpoints", policy_name)

        print(f"\nTreinando e analisando para policy: {policy_name}")

        mean_return, agent, results_counter = RLTrainingUtils.run_training(
            agent_type=training_cfg.agent_type,
            actions=[0, 1],
            params=params,
            episodes=training_cfg.episodes,
            seed=training_cfg.seed,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=0,
            track_results=True,
            env_name=training_cfg.env_name,
            max_steps_per_episode=training_cfg.max_steps_per_episode,
            early_stopping=training_cfg.early_stopping,
            early_stopping_patience=training_cfg.early_stopping_patience,
            early_stopping_delta=training_cfg.early_stopping_delta,
            save_best=training_cfg.save_best,
            verbose=training_cfg.verbose
        )

        RLTrainingUtils._update_learning_curves(agent, policy_name, report_meta.window, report_meta.learning_curves)
        total_episodes = report_meta.config["training"]["episodes"]
        win_rate = results_counter["win"] / total_episodes if total_episodes > 0 else 0.0

        print(f"Retorno médio: {mean_return:.4f}")
        print(f"Vitórias: {results_counter['win']} | Derrotas: {results_counter['loss']} | Empates: {results_counter['draw']} | Taxa de vitória: {win_rate:.4f}")

        RLTrainingUtils.analyze_training(agent, save_dir=policy_dir, policy_name=None)

        RLTrainingUtils._save_metrics_txt(policy_dir, mean_return, results_counter, win_rate, params, report_meta.seeds_scores, report_meta.mean_score, report_meta.std_score)
        RLTrainingUtils.save_policy_metrics_html(
            policy_name=policy_name,
            mean_return=mean_return,
            results_counter=results_counter,
            params=params,
            seeds_scores=report_meta.seeds_scores,
            mean_score=report_meta.mean_score,
            std_score=report_meta.std_score,
            save_dir=policy_dir,
            win_rate=win_rate
        )

    @staticmethod
    def train_and_report_random(config, output_dir, label, window, env_name, seed, learning_curves):
        print("\nTreinando e analisando baseline: RandomAgent")
        random_policy_dir = os.path.join(output_dir, f"{label}_random") if label else os.path.join(output_dir, "random")
        mean_return, agent, results_counter = RLTrainingUtils.run_training(
            agent_type="RandomAgent",
            actions=[0, 1],
            params={},
            episodes=config["training"]["episodes"],
            seed=seed,
            checkpoint_dir=os.path.join("output", "checkpoints", f"{label}_random") if label else os.path.join("output", "checkpoints", "random"),
            checkpoint_interval=0,
            track_results=True,
            env_name=env_name
        )
        total_episodes = config["training"]["episodes"]
        win_rate = results_counter["win"] / total_episodes if total_episodes > 0 else 0.0

        RLTrainingUtils._update_learning_curves(agent, "random", window, learning_curves)
        RLTrainingUtils.analyze_training(agent, save_dir=random_policy_dir, policy_name=None)
        RLTrainingUtils._save_metrics_txt(random_policy_dir, mean_return, results_counter, win_rate, "RandomAgent", [], mean_return, 0.0)
        RLTrainingUtils.save_policy_metrics_html(
            policy_name="random",
            mean_return=mean_return,
            results_counter=results_counter,
            params={},
            seeds_scores=[],
            mean_score=mean_return,
            std_score=0.0,
            save_dir=random_policy_dir,
            win_rate=win_rate
        )

    @staticmethod
    def _update_learning_curves(agent, policy_name, window, learning_curves):
        if hasattr(agent, "episode_returns"):
            returns = agent.episode_returns
            if len(returns) >= window:
                moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
            else:
                moving_avg = np.array(returns)
            learning_curves[policy_name] = moving_avg

    @staticmethod
    def _save_metrics_txt(policy_dir, mean_return, results_counter, win_rate, params, seeds_scores, mean_score, std_score):
        with open(os.path.join(policy_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"Retorno médio (primeiro seed): {mean_return:.4f}\n")
            f.write(f"Vitórias: {results_counter['win']}\n")
            f.write(f"Derrotas: {results_counter['loss']}\n")
            f.write(f"Empates: {results_counter['draw']}\n")
            f.write(f"Taxa de vitória: {win_rate:.4f}\n")
            f.write(f"Parâmetros: {params}\n")
            f.write(f"Retornos de todos os seeds: {seeds_scores}\n")
            f.write(f"Média dos seeds: {mean_score:.4f}\n")
            f.write(f"Desvio padrão dos seeds: {std_score:.4f}\n")

    @staticmethod
    def save_policy_metrics_html(
        policy_name: str,
        mean_return: float,
        results_counter: dict,
        params: dict,
        seeds_scores: list,
        mean_score: float,
        std_score: float,
        save_dir: str,
        win_rate: float = None
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
                        <tr><th>Taxa de vitória</th><td>{win_rate:.4f}</td></tr>
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
    def run_training_multiple_seeds(
        agent_type: str,
        actions: List[int],
        params: Dict[str, Any],
        episodes: int,
        seeds: List[int],
        checkpoint_dir: str,
        checkpoint_interval: int = 5000,
        env_name: str = "Blackjack-v1",
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[float], List[Any], List[Dict[str, int]]]:
        mean_returns = []
        agents = []
        results_counters = []
        for seed in seeds:
            local_config = dict(config) if config else {}
            mean_return, agent, results_counter = RLTrainingUtils.run_training(
                agent_type=agent_type,
                actions=actions,
                params=params,
                episodes=episodes,
                seed=seed,
                checkpoint_dir=os.path.join(checkpoint_dir, f"seed_{seed}"),
                checkpoint_interval=checkpoint_interval,
                track_results=local_config.get("track_results", False),
                env_name=local_config.get("env_name", env_name),
                max_steps_per_episode=local_config.get("max_steps_per_episode", 100),
                early_stopping=local_config.get("early_stopping", False),
                early_stopping_patience=local_config.get("early_stopping_patience", 1000),
                early_stopping_delta=local_config.get("early_stopping_delta", 0.01),
                save_best=local_config.get("save_best", True),
                verbose=local_config.get("verbose", True)
            )
            mean_returns.append(mean_return)
            agents.append(agent)
            results_counters.append(results_counter)
        return mean_returns, agents, results_counters
