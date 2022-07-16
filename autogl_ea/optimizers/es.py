import random

import inspyred
import time

from autogl_ea.optimizers import HPOptimizer
from autogl_ea.utils import EASupport
from autogl_ea.utils import SearchSpaceBounder


class ES(HPOptimizer):
    """
    (μ/ρ,λ)-ES Evolution Strategy:
        - μ denotes the number of parents,
        - ρ ≤ μ the number of parents involved in the producing a single offspring (mixing number),
        - λ the number of offspring,
        - comma selection strategy.

    When we use a comma strategy we forget the previous solutions: every time we replace the μ parents, with
    the λ offspring, and from the λ offspring we generate the new parents. So we completely forget
    the previous parents. On the one hand, this allows us to remove bad solutions, forcing the algorithm to be
    less exploitative and more explorative.
    This could be useful in the cases in which we’ve a moving optimum, and we need to forget the previous
    better solutions.
    """

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

        print('\nRunning ES...')
        super().optimize(trainer, dataset, time_limit, memory_limit)

        rand = random.Random()
        rand.seed(int(time.time()))

        ea = inspyred.ec.ES(rand)
        ea.terminator = [inspyred.ec.terminators.evaluation_termination,
                         inspyred.ec.terminators.diversity_termination]

        ea_support = EASupport(self.current_space)
        pop_generator = ea_support.generate_initial_population
        ssb = SearchSpaceBounder(self.current_space)

        final_pop = ea.evolve(generator=pop_generator,
                              #
                              evaluator=self.evaluate_candidates,
                              # Mu parameter, as defined in Bu et al.'s paper.
                              pop_size=10,
                              #
                              bounder=ssb,
                              #
                              max_generations=2)

        return self.post_Inspyred_optimization(final_pop)
