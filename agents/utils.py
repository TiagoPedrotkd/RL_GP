import numpy as np
import matplotlib.pyplot as plt
import io
import base64

class AgentVisualizer:
    Q_TABLE_EMPTY_MSG = "Q-table is empty."

    @staticmethod
    def plot_learning_curve(episode_returns, window=100, title="Learning Curve", save_path=None, show=False):
        """
        Plot moving average of episode returns.
        """

        if not episode_returns or len(episode_returns) < window:
            print("Not enough data to plot learning curve.")
            return None

        returns = np.array(episode_returns)
        moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')

        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg, label=f"{window}-episode moving average")
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return save_path
        
        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_base64}"/>'

    @staticmethod
    def plot_q_table(agent, title="Q-table Heatmap", save_path=None, show=False):
        """
        Visualize Q-table as a heatmap (for discrete state/action spaces).
        Only works if state and action spaces are small and discrete.
        """

        states = list(agent.Q.keys())

        if not states:
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return None
        
        try:
            np.array(states).max(axis=0) + 1
        except Exception:
            print("Cannot infer state shape for Q-table visualization.")
            return None
        
        q_matrix = np.zeros((len(states), len(agent.actions)))
        for i, state in enumerate(states):
            q_matrix[i] = agent.Q[state]

        plt.figure(figsize=(10, 6))
        plt.imshow(q_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Q-value')
        plt.xlabel('Action')
        plt.ylabel('State index')
        plt.title(title)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return save_path
        
        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_base64}"/>'

    @staticmethod
    def plot_policy(agent, title="Policy (Best Action per State)", save_path=None, show=False):
        """
        Visualize the greedy policy as a bar plot (for small discrete state spaces).
        """

        states = list(agent.Q.keys())

        if not states:
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return None
        
        try:
            best_actions = [np.argmax(agent.Q[state]) for state in states]
        except Exception:
            print("Cannot extract best actions for policy visualization.")
            return None
        
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(states)), best_actions)
        plt.xlabel("State index")
        plt.ylabel("Best Action")
        plt.title(title)
        plt.grid(True, axis='y')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return save_path
        
        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_base64}"/>'

    @staticmethod
    def plot_q_value_histogram(agent, title="Q-value Distribution", save_path=None, show=False):
        """
        Plot a histogram of all Q-values in the agent's Q-table.
        """

        q_values = []
        for q_arr in agent.Q.values():
            q_values.extend(q_arr if hasattr(q_arr, '__iter__') and not isinstance(q_arr, str) else [q_arr])

        if not q_values:
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return None
        
        plt.figure(figsize=(8, 4))
        plt.hist(q_values, bins=30, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel("Q-value")
        plt.ylabel("Frequency")
        plt.grid(True, axis='y')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return save_path
        
        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_base64}"/>'

    @staticmethod
    def save_all_visualizations(agent, prefix="agent", window=100):
        """
        Save all visualizations as PNG files.
        Returns a dict with file paths.
        """

        files = {}
        files['learning_curve'] = AgentVisualizer.plot_learning_curve(agent.episode_returns, window=window, save_path=f"{prefix}_learning_curve.png")
        files['q_table'] = AgentVisualizer.plot_q_table(agent, save_path=f"{prefix}_q_table.png")
        files['policy'] = AgentVisualizer.plot_policy(agent, save_path=f"{prefix}_policy.png")
        files['q_hist'] = AgentVisualizer.plot_q_value_histogram(agent, save_path=f"{prefix}_q_hist.png")
        return files

    @staticmethod
    def generate_html_report(agent, agent_name="Agent", window=100):
        """
        Generate an HTML report with learning curve, Q-table, policy, and Q-value histogram.
        """

        html = f"<h2>{agent_name} Report</h2>"

        if hasattr(agent, "episode_returns") and agent.episode_returns:
            html += "<h3>Learning Curve</h3>"
            img = AgentVisualizer.plot_learning_curve(agent.episode_returns, window=window)
            if img:
                html += img

        html += "<h3>Q-table Visualization</h3>"
        img = AgentVisualizer.plot_q_table(agent, title=f"{agent_name} Q-table")
        if img:
            html += img

        html += "<h3>Policy Visualization</h3>"
        img = AgentVisualizer.plot_policy(agent, title=f"{agent_name} Policy")
        if img:
            html += img

        html += "<h3>Q-value Distribution</h3>"
        img = AgentVisualizer.plot_q_value_histogram(agent, title=f"{agent_name} Q-value Distribution")
        if img:
            html += img

        return html
