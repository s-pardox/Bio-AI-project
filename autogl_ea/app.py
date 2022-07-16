import torch

from autogl_ea.settings import search_space as ss

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier

from autogl_ea.optimizers import *


def launch(alg, dataset='cora', graph_model=['gcn']):
    if alg == 'GA':
        optimizer = GA()
    elif alg == 'PSO':
        optimizer = PSO()
    elif alg == 'DEA':
        optimizer = DEA()
    elif alg == 'ES':
        optimizer = ES()
    elif alg == 'CMA-ES':
        optimizer = CMA_ES()

    dataset = build_dataset_from_name(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    __solve(optimizer, dataset, graph_model, device)


def __solve(optimizer, dataset, graph_model, device):
    # Take a look to autogl/solver/base.py and, in particular, to node_classifier.py, in the same folder (the file
    # contains the AutoNodeClassifier class, that technically is labeled as 'solver').
    solver = AutoNodeClassifier(

        # The (name of) the trainer used in this solver. Default to ``NodeClassificationFull``.
        # Take a look to autogl/module/train/node_classification_full.py
        default_trainer='NodeClassificationFull',

        # We can bypass it, for the moment.
        # feature_module = 'deepgl',

        # graph_models = ['gcn', 'gat'],
        graph_models=graph_model,

        # Let's use our own HPO module :-)
        hpo_module=optimizer,

        # We can bypass it, for the moment.
        # ensemble_module = 'voting',

        #
        device=device,
        #
        max_evals=5,

        # https://autogl.readthedocs.io/en/latest/docfile/tutorial/t_hpo.html#search-space
        # The following trainer's parameters (valued as the ones defined in Bu et al.'s paper) are passed to
        # AutoNodeClassifier (node_classifier.py), overwriting the default ones.
        trainer_hp_space=ss.SEARCH_SPACE['trainer_hp_space'],

        # The following trainer's parameters (valued as the ones defined in Bu et al.'s paper) are passed to
        # the GCN model (autogl/module/model/encoders/_dgl/_gcn.py), passing through the AutoNodeClassifier
        # (node_classifier.py) solver, overwriting the default ones.
        model_hp_spaces=ss.SEARCH_SPACE['model_hp_space']
    )

    # As default behavior, splits 0.2 of total nodes/graphs for train and 0.4 of nodes/graphs for validation,
    # the rest 0.4 is left for test.
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