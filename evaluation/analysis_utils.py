import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import os

class EvaluationReport:
    """
    Classe utilitária para análise e exportação de resultados de agentes RL.
    """

    @staticmethod
    def plot_returns(agent_returns, window=100, title="Learning Curve", show=False):
        """
        Plota a curva de aprendizagem (média móvel dos retornos).
        Retorna uma tag <img> HTML com o gráfico em base64.
        """

        plt.figure(figsize=(10, 5))
        for agent_name, returns in agent_returns.items():
            avg = np.convolve(returns, np.ones(window)/window, mode="valid")
            plt.plot(avg, label=agent_name)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(f"Return (Moving Avg over {window})")
        plt.legend()
        plt.grid()
        if show:
            plt.show()
            return None
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_base64}"/>'

    @staticmethod
    def plot_final_return_distributions(agent_rewards, show=False):
        """
        Plota a distribuição dos retornos finais dos agentes.
        Retorna uma tag <img> HTML com o gráfico em base64.
        """

        plt.figure(figsize=(8, 5))
        plt.boxplot(agent_rewards.values(), labels=agent_rewards.keys())
        plt.title("Return Distribution After Training")
        plt.ylabel("Return")
        plt.grid()
        if show:
            plt.show()
            return None
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_base64}"/>'

    @staticmethod
    def print_summary(agent_rewards):
        """
        Imprime e retorna uma tabela HTML com estatísticas resumo dos agentes.
        """

        html = "<table border='1'><tr><th>Agent</th><th>Mean</th><th>Std</th></tr>"
        for name, rewards in agent_rewards.items():
            mean = np.mean(rewards)
            std = np.std(rewards)
            print(f"{name}: {mean:.2f} ± {std:.2f}")
            html += f"<tr><td>{name}</td><td>{mean:.2f}</td><td>{std:.2f}</td></tr>"
        html += "</table>"

        return html

    @staticmethod
    def export_results_to_html(agent_rewards, filename="results_summary.html", report_dir="report"):
        """
        Exporta estatísticas e gráficos para um ficheiro HTML na pasta especificada.
        """

        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        filepath = os.path.join(report_dir, filename)

        html = "<html><head><title>RL Agent Results</title></head><body>"
        html += "<h2>Summary Statistics</h2>"
        html += EvaluationReport.print_summary(agent_rewards)
        html += "<h2>Learning Curves</h2>"
        html += EvaluationReport.plot_returns(agent_rewards)
        html += "<h2>Final Return Distributions</h2>"
        html += EvaluationReport.plot_final_return_distributions(agent_rewards)
        html += "</body></html>"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
            
        print(f"Results exported to {filepath}")

