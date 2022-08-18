import argparse
import yaml
from autogl_ea.settings import config as cfg

"""This module must be called before the actual run of the AutoGL-EA algorithm. It sets the hyperparameters of the 
evolutionary algorithm, that are eventually used by WandB as configuration values. This will help to perform 
a-posteriori analysis. """


def insert_data_rec(iterable, search_key, data):
    if isinstance(iterable, list):
        for item in iterable:
            if isinstance(item, (list, dict)):
                insert_data_rec(item, search_key, data)

    elif isinstance(iterable, dict):
        for k, v in iterable.items():
            if k == search_key:
                iterable[k].update(data)
            if isinstance(v, (list, dict)):
                insert_data_rec(v, search_key, data)


def check_errors(args):
    assert args.alg in cfg.AVAILABLE_ALG.keys(), 'Bio-AI algorithm not found.'
    assert args.pop_size is not None, 'Population Size must be set to an integer value'

    if args.alg == 'GA':
        assert args.max_eval is not None, 'Maximum number of candidate solutions Evaluations must be set to an ' \
                                          'integer value '
        assert args.cr_rate is not None, 'Crossover Rate must be set to a float value'
        assert args.mu_rate is not None, 'Mutation Rate must be set to a float value'
    elif args.alg == 'DE':
        assert args.max_gen is not None, 'Maximum number of Generations must be set to an integer value'
        assert args.cr_rate is not None, 'Crossover Rate must be set to a float value'
        assert args.mu_rate is not None, 'Mutation Rate must be set to a float value'
    elif args.alg in ['ES_plus', 'ES_comma']:
        assert args.max_eval is not None, 'Maximum number of candidate solutions Evaluations must be set to an ' \
                                          'integer value '
    elif args.alg == 'CMA-ES':
        assert args.max_gen is not None, 'Maximum number of Generations must be set to an integer value'
    else:
        assert args.max_eval is not None, 'Maximum number of candidate solutions Evaluations must be set to an ' \
                                          'integer value '
        assert args.inertia is not None, 'Inertia velocity must be set to a float value'
        assert args.cognitive is not None, 'Cognitive velocity must be set to a float value'
        assert args.social is not None, 'Social velocity must be set to a float value'


def main():
    parser = argparse.ArgumentParser(description='AutoGL-EA')
    parser.add_argument('-alg', type=str, default='GA',
                        help='Options: GA, PSO, DE, ES_plus, ES_comma, CMA-ES; default=GA')
    parser.add_argument('-max_eval', type=int, default=None, help='Set the max number of evaluations')
    parser.add_argument('-max_gen', type=int, default=None, help='Set the max number of generations')
    parser.add_argument('-pop_size', type=int, default=None, help='Set the population size')
    parser.add_argument('-cr_rate', type=float, default=None, help='Set the crossover rate')
    parser.add_argument('-mu_rate', type=float, default=None, help='Set the mutation rate')
    parser.add_argument('-inertia', type=float, default=None, help='Set the Inertia velocity (only for PSO)')
    parser.add_argument('-cognitive', type=float, default=None, help='Set the Cognitive velocity (only for PSO)')
    parser.add_argument('-social', type=float, default=None, help='Set the Social velocity (only for PSO)')

    args = parser.parse_args()

    check_errors(args)

    with open(cfg.EA_HP_PATH) as yml_file:
        data = yaml.safe_load(yml_file)

    insert_data_rec(data, search_key='max_eval', data={'value': args.max_eval})
    insert_data_rec(data, search_key='max_gen', data={'value': args.max_gen})
    insert_data_rec(data, search_key='pop_size', data={'value': args.pop_size})
    insert_data_rec(data, search_key='crossover_rate', data={'value': args.cr_rate})
    insert_data_rec(data, search_key='mutation_rate', data={'value': args.mu_rate})
    insert_data_rec(data, search_key='inertia_v', data={'value': args.inertia})
    insert_data_rec(data, search_key='cognitive_v', data={'value': args.cognitive})
    insert_data_rec(data, search_key='social_v', data={'value': args.social})

    with open(cfg.EA_HP_PATH, 'w') as yml_file:
        yaml.safe_dump(data, yml_file)


if __name__ == '__main__':
    main()
