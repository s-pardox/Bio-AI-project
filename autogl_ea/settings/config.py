import os

# os.getcwd() gets as root directory the same folder in which the main run script (run.py) is located

AVAILABLE_ALG = {
    'GA': 'Genetic Algorithm',
    'PSO': 'Particle Swarm Optimization',
    'DE': 'Differential Evolution',
    'ES_plus': '(μ + λ) Evolution Strategy',
    'ES_comma': '(μ, λ) Evolution Strategy',
    'CMA-ES': 'Covariance Matrix Adaptation - Evolution Strategy'}

# This is exactly the WandB default config file path.
EA_HP_PATH = os.getcwd() + '/config-defaults.yaml'
