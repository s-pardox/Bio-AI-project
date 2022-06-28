import torch
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cora_dataset = build_dataset_from_name('cora')

solver = AutoNodeClassifier(
    feature_module=None,
    graph_models=['gcn'],
    hpo_module='cmaes',
    ensemble_module=None,
    device=device
)


solver.fit(cora_dataset, time_limit=3600)
solver.get_leaderboard().show()


