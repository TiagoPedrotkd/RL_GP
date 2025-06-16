import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict

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
        Visualize Q-table as a heatmap.
        - For Blackjack: state-action dict (discrete).
        - For Pendulum: 3D numpy array (angle_bin, vel_bin, action_idx).
        """
        if not hasattr(agent, "Q"):
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return ""

        Q = agent.Q

        # Convert defaultdict to dict for visualization
        if isinstance(Q, defaultdict):
            Q = dict(Q)

        AgentVisualizer._print_q_table_summary(Q)

        fig = None
        if isinstance(Q, np.ndarray) and Q.ndim == 3:
            fig = AgentVisualizer._plot_q_table_ndarray(Q, title)
        elif isinstance(Q, dict) or hasattr(Q, "keys"):
            fig = AgentVisualizer._plot_q_table_dict(Q, agent, title)
            if fig is None:
                return ""
        else:
            print("Cannot visualize Q-table: unknown Q type.")
            return ""

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            return save_path

        if show:
            plt.show()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}"/>'

    @staticmethod
    def _print_q_table_summary(q):
        print("=== Q-table summary ===")
        if isinstance(q, np.ndarray):
            print(f"Q-table type: ndarray, shape: {q.shape}, min: {np.min(q)}, max: {np.max(q)}")
        elif isinstance(q, dict) or hasattr(q, "keys"):
            print(f"Q-table type: dict, number of states: {len(q)}")
            if len(q) > 0:
                sample_state = next(iter(q))
                print(f"Sample state: {sample_state}, Q-values: {q[sample_state]}")
        else:
            print("Q-table type: unknown")

    @staticmethod
    def _plot_q_table_ndarray(q, title):
        q_max = np.max(q, axis=2)
        fig = plt.figure(figsize=(10, 6))
        im = plt.imshow(q_max, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im, label='Max Q-value')
        plt.xlabel('Velocity bin')
        plt.ylabel('Angle bin')
        plt.title(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def _plot_q_table_dict(q, agent, title):
        states = list(q.keys())
        if not states:
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return None
        try:
            states = sorted(states)
        except Exception:
            pass
        q_matrix = np.zeros((len(states), len(agent.actions)))
        for i, state in enumerate(states):
            q_matrix[i] = q[state]
        fig = plt.figure(figsize=(10, 6))
        im = plt.imshow(q_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Q-value')
        plt.xlabel('Action')
        plt.ylabel('State index')
        plt.title(title)
        plt.tight_layout()
        return fig


    @staticmethod
    def plot_policy(agent, title="Policy (Best Action per State)", save_path=None, show=False):
        """
        Visualize the greedy policy:
        - For Blackjack: bar plot of best action per state.
        - For Pendulum: 2D heatmap of best action index (angle_bin x vel_bin).
        """
        if hasattr(agent, "Q"):
            Q = agent.Q
        else:
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return ""

        if isinstance(Q, np.ndarray) and Q.ndim == 3:
            best_actions = np.argmax(Q, axis=2)
            plt.figure(figsize=(10, 6))
            im = plt.imshow(best_actions, aspect='auto', cmap='tab20', origin='lower')
            plt.colorbar(im, label='Best Action Index')
            plt.xlabel('Velocity bin')
            plt.ylabel('Angle bin')
            plt.title(title)
            plt.tight_layout()

        elif isinstance(Q, dict) or hasattr(Q, "keys"):
            states = list(Q.keys())
            if not states:
                print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
                return ""
            try:
                best_actions = [np.argmax(Q[state]) for state in states]
            except Exception:
                print("Cannot extract best actions for policy visualization.")
                return ""
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(states)), best_actions)
            plt.xlabel("State index")
            plt.ylabel("Best Action")
            plt.title(title)
            plt.grid(True, axis='y')
            plt.tight_layout()
        else:
            print("Cannot visualize policy: unknown Q type.")
            return ""

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
    def _flatten_q_values(q):
        """
        Helper to flatten Q-table values into a 1D numpy array.
        """
        if isinstance(q, np.ndarray):
            return q.flatten()
        elif isinstance(q, dict) or hasattr(q, "values"):
            q_values = []
            for q_arr in q.values():
                if hasattr(q_arr, '__iter__') and not isinstance(q_arr, str):
                    q_values.extend(q_arr)
                else:
                    q_values.append(q_arr)
            return np.array(q_values)
        return None

    @staticmethod
    def plot_q_value_histogram(agent, title="Q-value Distribution", save_path=None, show=False):
        """
        Plot a histogram of all Q-values in the agent's Q-table.
        Works for both dict and numpy Q-tables.
        """
        if not hasattr(agent, "Q"):
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return ""

        Q = agent.Q
        q_values = AgentVisualizer._flatten_q_values(Q)
        if q_values is None or (isinstance(q_values, np.ndarray) and q_values.size == 0) or (not isinstance(q_values, np.ndarray) and not q_values):
            print(AgentVisualizer.Q_TABLE_EMPTY_MSG)
            return ""

        plt.figure(figsize=(8, 4))
        plt.hist(q_values, bins=30, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel("Q-value")
        plt.ylabel("Frequency")
        plt.grid(True, axis='y')
        plt.tight_layout()

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