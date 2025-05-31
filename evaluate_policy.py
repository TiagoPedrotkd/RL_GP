import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import random

from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.monte_carlo import MonteCarloAgent
from agents.random_agent import RandomAgent
from environments.blackjack_env import make_blackjack_env

def load_agent(agent_type, actions, qtable_path, params, seed=42):
    agent_type_lower = agent_type.lower()
    if agent_type_lower == "qlearningagent":
        agent = QLearningAgent(actions, **params, seed=seed)
    elif agent_type_lower == "sarsaagent":
        agent = SARSAAgent(actions, **params, seed=seed)
    elif agent_type_lower == "montecarloagent":
        agent = MonteCarloAgent(actions, **params, seed=seed)
    elif agent_type_lower == "randomagent":
        agent = RandomAgent(actions, seed=seed)
        return agent
    else:
        raise ValueError(f"Agente não suportado: {agent_type}")
    agent.load(qtable_path)
    return agent

def evaluate_agent(agent, env, n_episodes=1000):
    returns = []
    win, loss, draw = 0, 0, 0
    for _ in range(n_episodes):
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        done = False
        ep_return = 0
        while not done:
            action = agent.policy(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            ep_return += reward
            state = next_state
        returns.append(ep_return)
        if ep_return > 0:
            win += 1
        elif ep_return < 0:
            loss += 1
        else:
            draw += 1
    stats = {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "win": win,
        "loss": loss,
        "draw": draw,
        "win_rate": win / n_episodes if n_episodes > 0 else 0.0,
        "returns": returns
    }
    return stats

def plot_returns_histogram(returns, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribuição dos Retornos por Episódio")
    plt.xlabel("Retorno")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Avaliação de agente treinado no Blackjack.")
    parser.add_argument("--agent_type", type=str, required=True, help="Tipo de agente: QLearningAgent, SARSAAgent, MonteCarloAgent, RandomAgent")
    parser.add_argument("--qtable_path", type=str, required=False, help="Caminho para o ficheiro da Q-table (npy)")
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episódios de avaliação")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha (apenas para QLearning/SARSA)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output_dir", type=str, default="output/eval", help="Diretório para guardar resultados")
    args = parser.parse_args()

    # Set global seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    actions = [0, 1]
    params = {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}

    env = make_env(env_name=args.agent_type if hasattr(args, "env_name") else "Blackjack-v1", seed=args.seed)
    agent = load_agent(args.agent_type, actions, args.qtable_path, params, seed=args.seed)

    stats = evaluate_agent(agent, env, n_episodes=args.episodes)

    print(f"Retorno médio: {stats['mean_return']:.4f}")
    print(f"Vitórias: {stats['win']} | Derrotas: {stats['loss']} | Empates: {stats['draw']}")
    print(f"Taxa de vitória: {stats['win_rate']:.4f}")

    # Salva estatísticas
    with open(os.path.join(args.output_dir, "eval_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Retorno médio: {stats['mean_return']:.4f}\n")
        f.write(f"Desvio padrão: {stats['std_return']:.4f}\n")
        f.write(f"Vitórias: {stats['win']}\n")
        f.write(f"Derrotas: {stats['loss']}\n")
        f.write(f"Empates: {stats['draw']}\n")
        f.write(f"Taxa de vitória: {stats['win_rate']:.4f}\n")

    # Salva retornos
    np.save(os.path.join(args.output_dir, "eval_returns.npy"), np.array(stats["returns"]))

    # Gráfico
    plot_returns_histogram(stats["returns"], save_path=os.path.join(args.output_dir, "eval_returns_hist.png"))

if __name__ == "__main__":
    main()
