import torch
import wandb

from autogl.datasets import build_dataset_from_name
from autogl.solver import AutoNodeClassifier, AutoGraphClassifier

from autogl_ea.settings import search_space as ss
from autogl_ea.utils import SearchSpaceMng
from autogl_ea.optimizers import *


def launch(alg='GA', dataset='cora', graph_model=None, hidden_layers=1, problem='node'):
    if alg == 'GA':
        optimizer = GA()
    elif alg == 'PSO':
        optimizer = PSO()
    elif alg == 'DE':
        optimizer = DE()
    elif alg == 'ES':
        optimizer = ES()
    elif alg == 'CMA-ES':
        optimizer = CMA_ES()

    if graph_model is None:
        graph_model = ['gcn']

    dataset = build_dataset_from_name(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Eventually modifies the search space, altering the number of hidden layers in the NN.
    ss_mng = SearchSpaceMng(ss.SEARCH_SPACE)
    search_space = ss_mng.modify_ss_by_hl(hidden_layers)

    __solve(optimizer, dataset, graph_model, search_space, device, problem)


def __solve(optimizer, dataset, graph_model, search_space, device, problem):
    # Take a look to autogl/solver/base.py and, in particular, to node_classifier.py, in the same folder (the file
    # contains the AutoNodeClassifier class, that technically is labeled as 'solver').

    if problem == 'node':

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
            trainer_hp_space=search_space['trainer_hp_space'],

            # The following trainer's parameters (valued as the ones defined in Bu et al.'s paper) are passed to
            # the GCN model (autogl/module/model/encoders/_dgl/_gcn.py), passing through the AutoNodeClassifier
            # (node_classifier.py) solver, overwriting the default ones.
            model_hp_spaces=search_space['model_hp_space']
        )

        solver.fit(dataset, time_limit=120)

    elif problem == 'graph':

        solver = AutoGraphClassifier(
            default_trainer='GraphClassificationFull',
            # graph_models=graph_model,
            graph_models=['gin'],
            hpo_module=optimizer,
            device=device,
            max_evals=5,
            trainer_hp_space=search_space['trainer_hp_space'],
            model_hp_spaces=search_space['model_hp_space']
        )

        solver.fit(dataset, time_limit=120, train_split=0.8, val_split=0.2)


    # As default behavior, splits 0.2 of total nodes/graphs for train and 0.4 of nodes/graphs for validation,
    # the rest 0.4 is left for test.
    #
    # time_limit: int
    # The time limit of the whole fit process (in seconds). If set below 0, will ignore time limit. Default ``-1``.

    """
    # get current leaderboard of the solver
    # lb=solver.get_leaderboard()
    # show the leaderboard info
    # lb.show()
    """

    acc = solver.evaluate(metric='acc')

    print('\nTest accuracy: {:.4f}'.format(acc))
    wandb.log({'test_acc:': float(acc)})

    best_individual = solver.hpo_module.best_para
    print('\nBest individual (encoded):\n{}'.format(best_individual))

    named_individual = solver.hpo_module.gen_named_individual(best_individual)
    print('\nBest named individual (encoded):\n{}'.format(named_individual))

    print('\nAutoGL best parameters for trainer (decoded):\n{}'.format(optimizer.best_trainer.hyper_parameters))
    print('\nAutoGL best parameters for model (decoded):\n{}'.format(optimizer.best_trainer.model.hyper_parameters))

    print('\nFinal population diversity:\n{}'.format(solver.hpo_module.diversity))
