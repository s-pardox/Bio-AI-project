"""
import torch
from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier
from autogl.module.train import Acc

cora_dataset = build_dataset_from_name('cora')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

solver = AutoNodeClassifier(
    feature_module='deepgl',
    graph_models=['gcn', 'gat'],
    hpo_module='anneal',
    ensemble_module='voting',
    device=device
)

solver.fit(cora_dataset, time_limit=3600)

predicted = solver.predict_proba()
print('Test accuracy: ', Acc.evaluate(predicted,
                                      cora_dataset.data.y[cora_dataset.data.test_mask].cpu().numpy()))
"""

import torch

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier

from TestOptimizer import TestOptimizer

dataset = build_dataset_from_name('cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# autoClassifier = AutoNodeClassifier()
solver = AutoNodeClassifier(
    # feature_module='deepgl',
    # graph_models=['gcn', 'gat'],
    graph_models=['gcn'],
    # hpo_module='anneal',
    hpo_module=TestOptimizer(),
    ensemble_module='voting',
    device=device,
    max_evals=5
)

# Split 0.2 of total nodes/graphs for train and 0.4 of nodes/graphs for validation,
# the rest 0.4 is left for test.
# time_limit: int
# The time limit of the whole fit process (in seconds). If set below 0,
# will ignore time limit. Default ``-1``.
solver.fit(dataset, time_limit=120)

# get current leaderboard of the solver
lb = solver.get_leaderboard()
# show the leaderboard info
lb.show()

acc = solver.evaluate(metric='acc')

print("test acc: {:.4f}".format(acc))
