import subprocess
import sys
import os

if __name__ == "__main__":

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    print("=== Treinando Blackjack ===")
    subprocess.run([sys.executable, os.path.join("src", "training", "train_blackjack.py")], check=True, env=env)
    print("\n=== Treinando Pendulum ===")
    subprocess.run([sys.executable, os.path.join("src", "training", "train_pendulum.py")], check=True, env=env)
    print("\nTreinamento de ambos os ambientes concluído.")
