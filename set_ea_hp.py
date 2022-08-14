import argparse
import yaml
from autogl_ea.settings import config as cfg

"""
This module must be called before the actual run of the AutoGL-EA algorithm. It sets the hyperparameters of the evolutionary
algorithm, that are eventually used by WandB as configuration values. This will help to perform a-posteriori analysis.
"""

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


def main():

    parser = argparse.ArgumentParser(description='AutoGL-EA')
    parser.add_argument('-pop_size', type=int, default=10, help='Set the population size')
    parser.add_argument('-cr_rate', type=float, default=0.7, help='Set the crossover rate')
    parser.add_argument('-mu_rate', type=float, default=0.2, help='Set the mutation rate')

    args = parser.parse_args()

    with open(cfg.EA_HP_PATH) as yml_file:
        data = yaml.safe_load(yml_file)

    insert_data_rec(data, search_key='crossover_rate', data={'value': args.cr_rate})
    insert_data_rec(data, search_key='mutation_rate', data={'value': args.mu_rate})
    insert_data_rec(data, search_key='pop_size', data={'value': args.pop_size})

    with open(cfg.EA_HP_PATH, 'w') as yml_file:
        yaml.safe_dump(data, yml_file)


if __name__ == '__main__':
    main()



