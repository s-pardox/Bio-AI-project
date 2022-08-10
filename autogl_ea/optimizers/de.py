import random
import yaml

import inspyred
import time

from autogl_ea.optimizers import HPOptimizer
from autogl_ea.utils import EASupport
from autogl_ea.utils import SearchSpaceBounder


class DE(HPOptimizer):
    """Differential Evolution"""

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

        print('\nRunning DE...')
        super().optimize(trainer, dataset, time_limit, memory_limit)

        rand = random.Random()
        rand.seed(int(time.time()))

        ea = inspyred.ec.DEA(rand)
        ea.observer = inspyred.ec.observers.stats_observer
        ea.terminator = inspyred.ec.terminators.evaluation_termination

        ea_support = EASupport(self.current_space, self.design_variables)
        pop_generator = ea_support.generate_initial_population
        ssb = SearchSpaceBounder(self.current_space)
        ea.observer = ea_support.observer

        config = self.getConfig()

        final_pop = ea.evolve(generator=pop_generator,
                              #
                              evaluator=self.evaluate_candidates,
                              # Population size.
                              pop_size=config['pop_size']['value'],
                              # Search Space bounder.
                              bounder=ssb,
                              #
                              max_generations=100,
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
