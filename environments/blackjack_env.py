"""
Módulo utilitário para criação do ambiente Blackjack-v1 do Gymnasium.
"""

import gymnasium as gym

def make_blackjack_env(seed: int = None):
    """
    Cria e retorna uma instância do ambiente Blackjack-v1 do Gymnasium.

    Parâmetros:
        seed (int, opcional): Semente para reprodutibilidade.

    Retorna:
        env: Ambiente Blackjack-v1 configurado.
    """

    env = gym.make("Blackjack-v1")

    if seed is not None:
        env.reset(seed=seed)
        
    return env
