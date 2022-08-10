import random
import yaml

import inspyred
import time

from autogl_ea.optimizers import HPOptimizer
from autogl_ea.utils import EASupport
from autogl_ea.utils import SearchSpaceBounder


class GA(HPOptimizer):
    """'Classic' Genetic Algorithm"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get('max_evals', 2)

        # Tells to the __fit method to not invert the performance result.
        self.maximize = False

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        """This method is automatically invoked by AutoGL; has to be considered the entry point of the optimization
        process.
        It inherits from HPOptimizer.
        """

        print('\nRunning GA...')
        super().optimize(trainer, dataset, time_limit, memory_limit)

        rand = random.Random()
        rand.seed(int(time.time()))

        ga = inspyred.ec.GA(rand)
        ga.observer = inspyred.ec.observers.stats_observer
        ga.terminator = inspyred.ec.terminators.evaluation_termination

        ea_support = EASupport(self.current_space, self.design_variables)
        pop_generator = ea_support.generate_initial_population
        ga.observer = ea_support.observer

        config = self.getConfig()

        final_pop = ga.evolve(evaluator=self.evaluate_candidates,
                              #
                              generator=pop_generator,
                              # Number of generations = max_evaluations / pop_size.
                              max_evaluations=10,
                              #
                              num_elites=0,
                              # Population size.
                              pop_size=config['pop_size']['value'],
                              # Number of individuals that have to be generated as initial population.
                              num_inputs=2,
                              crossover_rate=config['crossover_rate']['value'],
                              mutation_rate=config['mutation_rate']['value'])

        return self.post_Inspyred_optimization(final_pop)

    def getConfig(self):
        """This method gets the hyperparameters of the EA from the yaml file
        """

        config = dict()

        with open(r'config-defaults.yaml') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            config = params.copy()

        return config
