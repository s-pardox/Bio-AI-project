import random

import inspyred
import time

from autogl_ea.optimizers import HPOptimizer
from autogl_ea.utils import EASupport
from autogl_ea.utils import SearchSpaceBounder


class DEA(HPOptimizer):
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

        print('\nRunning DEA...')
        super().optimize(trainer, dataset, time_limit, memory_limit)

        rand = random.Random()
        rand.seed(int(time.time()))

        ea = inspyred.ec.DEA(rand)
        ea.terminator = inspyred.ec.terminators.evaluation_termination

        ea_support = EASupport(self.current_space)
        pop_generator = ea_support.generate_initial_population
        ssb = SearchSpaceBounder(self.current_space)

        final_pop = ea.evolve(generator=ea_support.generate_initial_population,
                              #
                              evaluator=pop_generator,
                              # Population size.
                              pop_size=25,
                              # Search Space bounder.
                              bounder=ssb,
                              #
                              max_generations=30)

        return self.post_Inspyred_optimization(final_pop)
