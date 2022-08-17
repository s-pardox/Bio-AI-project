import random

import inspyred
import time

from autogl_ea.optimizers import HPOptimizer
from autogl_ea.utils import EASupport
from autogl_ea.utils import SearchSpaceBounder


class ES(HPOptimizer):
    """
    Comment copied and pasted from inspyred/ec/ec.py

    Evolutionary computation representing a canonical evolution strategy.
    This class represents an evolution strategy which uses, by default, the default selection (i.e., all individuals
    are selected), an internal adaptive mutation using strategy parameters, and 'plus' replacement. It is expected that
    each candidate solution is a ``Sequence`` of real values.

    The candidate solutions to an ES are augmented by strategy parameters of the same length (using
    ``inspyred.ec.generators.strategize``). These strategy parameters are evolved along with the candidates and are
    used as the mutation rates for each element of the candidates. The evaluator is modified internally to use only the
    actual candidate elements (rather than also the strategy parameters), so normal evaluator functions may be used
    seamlessly.

    So, basically, the default Evolution Strategy implemented in Inspyred is:
        'mu plus lambda', i.e. self-adaptive (uncorrelated mutations with multiple sigma).

    It is possible to select the 'mu comma lambda' ES by passing parameter replacer='plus' to the constructor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get('max_evals', 2)

        # Tells to the __fit method to not invert the performance result.
        self.maximize = False
        # 'comma' or 'plus'.
        self.strategy = None

    def set_strategy(self, strategy):
        self.strategy = strategy

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        """This method is automatically invoked by AutoGL; has to be considered the entry point of the optimization
        process.
        It inherits from HPOptimizer.
        """

        print('\nRunning ES ({} strategy)...'.format(self.strategy))
        super().optimize(trainer, dataset, time_limit, memory_limit)

        rand = random.Random()
        rand.seed(int(time.time()))

        ea = inspyred.ec.ES(rand)
        ea.terminator = [inspyred.ec.terminators.evaluation_termination,
                         inspyred.ec.terminators.diversity_termination]

        # Otherwise, it uses the default 'plus' replacer.
        if self.strategy == 'comma':
            ea.replacer = inspyred.ec.replacers.comma_replacement

        ea_support = EASupport(self.current_space, self.design_variables)
        pop_generator = ea_support.generate_initial_population
        ssb = SearchSpaceBounder(self.current_space)

        final_pop = ea.evolve(generator=pop_generator,
                              #
                              evaluator=self.evaluate_candidates,
                              # Mu parameter, as defined in Bu et al.'s paper.
                              pop_size=4,
                              #
                              bounder=ssb,
                              #
                              max_generations=2)

        return self.post_Inspyred_optimization(final_pop)
