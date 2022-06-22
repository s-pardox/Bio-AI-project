import torch

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier

from TestOptimizer import TestOptimizer

dataset = build_dataset_from_name('cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Take a look to autogl/solver/base.py and, in particular, to node_classifier.py, in the same folder (the file contains
# the AutoNodeClassifier class, that technically is labeled as 'solver').
solver = AutoNodeClassifier(

    # The (name of) the trainer used in this solver. Default to ``NodeClassificationFull``.
    # Take a look to autogl/module/train/node_classification_full.py
    default_trainer='NodeClassificationFull',

    # We can bypass it, for the moment.
    # feature_module = 'deepgl',

    # graph_models = ['gcn', 'gat'],
    graph_models=['gcn'],

    # hpo_module = 'anneal',
    # Let's use our own HPO module :-)
    hpo_module=TestOptimizer(),

    # We can bypass it, for the moment.
    # ensemble_module = 'voting',

    device=device,
    max_evals=5,

    # https://autogl.readthedocs.io/en/latest/docfile/tutorial/t_hpo.html#search-space
    # The following trainer's parameters (valued as the ones defined in Bu et al.'s paper) are passed to
    # AutoNodeClassifier (node_classifier.py), overwriting the default ones.
    trainer_hp_space=[
        {
            # In Bu et al.'s paper: P1 - continuous param in the [0.01,0,05] range.
            'parameterName': 'lr',
            'type': 'DOUBLE',
            'maxValue': 1e-2,
            'minValue': 5e-2,
            'scalingType': 'LOG',
        },
        {
            # In Bu et al.'s paper: P2 - continuous param in the [0.0001,0.001] range.
            'parameterName': 'weight_decay',
            'type': 'DOUBLE',
            'maxValue': 1e-2,
            'minValue': 1e-4,
            'scalingType': 'LOG',
        },
        {
            # In Bu et al.'s paper: P5 - discrete param in the [100,300] range.
            'parameterName': 'max_epoch',
            'type': 'INTEGER',
            'maxValue': 300,
            'minValue': 100,
            'scalingType': 'LINEAR',
        },
        {
            # In Bu et al.'s paper: P6 - discrete param in the [10,30] range.
            'parameterName': 'early_stopping_round',
            'type': 'INTEGER',
            'maxValue': 30,
            'minValue': 10,
            'scalingType': 'LINEAR',
        }
    ],

    # The following trainer's parameters (valued as the ones defined in Bu et al.'s paper) are passed to
    # the GCN model (autogl/module/model/encoders/_dgl/_gcn.py), passing through the AutoNodeClassifier
    # (node_classifier.py) solver, overwriting the default ones.
    model_hp_spaces=[
        # 'encoder'
        [
            {
                # In Bu et al.'s paper: P3 - continuous param in the [0.2,0.8] range.
                'parameterName': 'dropout',
                'type': 'DOUBLE',
                'maxValue': 0.8,
                'minValue': 0.2,
                'scalingType': 'LINEAR',
            },
            {
                # This is exactly the default parameter defined in the GCN model itself.
                'parameterName': 'num_layers',
                'type': 'DISCRETE',
                'feasiblePoints': '2,3,4',
            },
            {
                # In Bu et al.'s paper: H1 - discrete param in the [4,16] range.
                # TO CLARIFY: Does it refer to the Number of Hidden Units _per layer_?
                # Yes, it does:
                #   [ "GCN, the number of layers in the convolution structure was fixed, and only the number of units in
                #   the hidden layer (H1) was adjusted." ]
                'parameterName': 'hidden',
                'type': 'NUMERICAL_LIST',
                'numericalType': 'INTEGER',
                'length': 3,
                'minValue': [4, 4, 4],
                'maxValue': [16, 16, 16],
                'scalingType': 'LINEAR',
                'cutPara': ('num_layers',),
                'cutFunc': lambda x: x[0] - 1,
            },
            {
                # In Bu et al.'s paper: P3 - discrete categorical param.
                'parameterName': 'act',
                'type': 'CATEGORICAL',
                'feasiblePoints': ['leaky_relu', 'relu', 'elu', 'tanh'],
            }
        ]
        # We don't have any 'decoder' counter-part parameters.
    ]
)

# Split 0.2 of total nodes/graphs for train and 0.4 of nodes/graphs for validation,
# the rest 0.4 is left for test.
# time_limit: int
# The time limit of the whole fit process (in seconds). If set below 0,
# will ignore time limit. Default ``-1``.
solver.fit(dataset, time_limit=120)

# get current leaderboard of the solver
# lb  =  solver.get_leaderboard()
# show the leaderboard info
# lb.show()

acc = solver.evaluate(metric='acc')

print('\ntest acc: {:.4f}'.format(acc))
