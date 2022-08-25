"""
UniTN Bio Inspired Artificial Intelligent final project - 2021/2022
Authors: Mattia Florio, Stefano Pardini

This module is the entry point of the program, it is responsible for parsing the command line parameters, launching
functions from autogl_ea.app package.
It has to be eventually executed after the execution of set_ea_hp.py module.
"""

import os
import argparse
import wandb

import autogl_ea.app as app
import autogl_ea.settings.wandb_settings
import autogl_ea.settings.config as cfg
from Analysis import save_best_decoded_individual


def main():
    """
    Command line arguments parser, WanDB initializer, launcher.
    """

    parser = argparse.ArgumentParser(description='AutoGL-EA')
    parser.add_argument('-alg', type=str, default='GA',
                        help='Options: GA, PSO, DE, ES_plus, ES_comma, CMA-ES; default=GA')
    parser.add_argument('-dataset', type=str, default='cora', help='Options: cora, proteins; default=cora')
    parser.add_argument('-graph_model', type=str, default='gcn', help='Options: gcn, gat; default=gcn')
    parser.add_argument('-hl', type=int, default=1, help='The number of hidden layers to be used; default=1')
    parser.add_argument('-problem', type=str, default='node', help='Classification options: node, graph; default=node')

    parser.add_argument('-wandb', type=bool, default=True, help='Log results on WandB; default=False')
    parser.add_argument('-wandb_group_name', type=str, default='Ste * Final Experiments',
                        help='WandB group name; default=Final Experiments')

    args = parser.parse_args()

    if args.wandb == 'False':
        # If you don't want your script to sync to the cloud.
        # https://docs.wandb.ai/guides/track/advanced/environment-variables
        os.environ['WANDB_MODE'] = 'offline'
    elif args.wandb == 'True':
        os.environ['WANDB_MODE'] = 'online'

    assert args.alg in cfg.AVAILABLE_ALG.keys(), 'Bio-AI algorithm not found.'
    assert args.dataset in ['cora', 'citeseer'], 'Dataset not found.'
    assert args.graph_model in ['gcn', 'gat'], 'Graph model not found.'
    assert 10 >= args.hl >= 1, 'Invalid number of hidden layers.'
    assert args.problem in ['node', 'graph'], 'Kind of problem not found.'

    # Parameters got from the command line parser.
    alg = args.alg
    dataset = args.dataset
    graph_model = args.graph_model
    hl = args.hl
    problem = args.problem
    wandb_group_name = args.wandb_group_name

    # You need to edit settings/wandb_settings.py, specifying WANDB_ENTITY (username), WANDB_API_KEY, etc.
    wandb_run_name = 'alg: {}, ds: {}, gm: {}, hl: {}, problem: {}'.format(alg, dataset, graph_model, hl, problem)

    # YAML config file.
    wandb.init(project='AutoGL-EA', name=wandb_run_name, entity='bio-ai-2022', group=wandb_group_name)

    # Command line launcher.
    test_acc, trainer, model = app.launch(alg=alg, dataset=dataset, graph_model=[graph_model], hidden_layers=hl,
                                          problem=problem)

    # Manual launcher.
    # test_acc, trainer, model = app.launch(alg='ES_comma', dataset='cora', graph_model=['gcn'], hidden_layers=1,
    #                                      problem='node')

    save_best_decoded_individual(wandb_group_name, test_acc, trainer, model)


if __name__ == '__main__':
    main()
