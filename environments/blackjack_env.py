"""
Módulo utilitário para criação do ambiente Blackjack-v1 do Gymnasium.
"""

import gymnasium as gym

def make_env(env_name="Blackjack-v1", seed=None):
    """
    Cria e retorna uma instância do ambiente Blackjack-v1 do Gymnasium.

    Parâmetros:
        env_name (str, opcional): Nome do ambiente a ser criado.
        seed (int, opcional): Semente para reprodutibilidade.

    Retorna:
        env: Ambiente Blackjack-v1 configurado.
    """

    env = gym.make(env_name)

    if seed is not None:
        env.reset(seed=seed)
        
    return env
