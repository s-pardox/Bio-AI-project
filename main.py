import torch

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier

from InspyredOptimizer import InspyredOptimizer

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

    # Let's use our own HPO module :-)
    # Available options: 'GA', 'PSO'.
    hpo_module=InspyredOptimizer(alg='PSO'),

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
            'maxValue': 5e-2,
            'minValue': 1e-2,
            'scalingType': 'LOG',
        },
        {
            # In Bu et al.'s paper: P2 - continuous param in the [0.0001,0.001] range.
            'parameterName': 'weight_decay',
            'type': 'DOUBLE',
            'maxValue': 1e-3,
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
                # TODO
                # We've temporarily fixed the number of layers to '2' (and consequently, to '1' the number of hidden
                # layers).
                'parameterName': 'num_layers',
                'type': 'FIXED',
                'value': 2,
            },
            {
                # In Bu et al.'s paper: H1 - discrete param in the [4,16] range.
                'parameterName': 'hidden',
                'type': 'NUMERICAL_LIST',
                'numericalType': 'INTEGER',
                # Has to be considered as 'max length'.
                'length': 1,
                'minValue': [4],
                'maxValue': [16],

                # TODO
                # Accordingly to Bu et al.'s paper, the values have to be transformed as ln(H1)
                # Does the LOG scale perform a 'ln' transformation? Yes, it does.
                'scalingType': 'LOG',

                # By expliciting 'cutPara' we force HPO to cut the list to a certain length which is dependent on
                # 'num_layers' param.
                'cutPara': ('num_layers',),
                # As general rule:
                #   len(hidden) = num_layers - 1
                'cutFunc': lambda x: x[0] - 1,
            },
            {
                # In Bu et al.'s paper: P3 - continuous param in the [0.2,0.8] range.
                'parameterName': 'dropout',
                'type': 'DOUBLE',
                'maxValue': 0.8,
                'minValue': 0.2,
                'scalingType': 'LINEAR',
            },
            {
                # In Bu et al.'s paper: P4 - discrete categorical param.
                'parameterName': 'act',
                'type': 'CATEGORICAL',
                'feasiblePoints': ['leaky_relu', 'relu', 'elu', 'tanh'],
            }
        ]
        # We don't have any 'decoder' counter-part parameters.
    ]
)

# As default behavior, splits 0.2 of total nodes/graphs for train and 0.4 of nodes/graphs for validation, the rest 0.4
# is left for test.
#
# time_limit: int
# The time limit of the whole fit process (in seconds). If set below 0, will ignore time limit. Default ``-1``.
solver.fit(dataset, time_limit=120)

"""
# get current leaderboard of the solver
# lb=solver.get_leaderboard()
# show the leaderboard info
# lb.show()
"""

acc = solver.evaluate(metric='acc')

print('\nTest accuracy: {:.4f}'.format(acc))
print('\nbest_para = \n{}'.format(solver.hpo_module.best_para))