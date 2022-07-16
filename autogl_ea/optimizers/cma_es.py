import random

import cma
import time

from autogl_ea.optimizers import HPOptimizer
from autogl_ea.utils import EASupport
from autogl_ea.utils import SearchSpaceBounder


class CMA_ES(HPOptimizer):
    """Covariance matrix adaptation evolution strategy"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get('max_evals', 2)

        # Tells to the __fit method to invert the performance result.
        self.maximize = True

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        """This method is automatically invoked by AutoGL; has to be considered the entry point of the optimization
        process.
        It inherits from HPOptimizer.
        """

        print('\nRunning CMA-ES...')
        super().optimize(trainer, dataset, time_limit, memory_limit)

        rand = random.Random()
        rand.seed(int(time.time()))

        # CMA-ES parameters.
        sigma = 0.5
        # λ: indicates the number of offspring produced.
        num_offspring = 100
        # μ: population size.
        pop_size = 20
        """!!! Very important constraint: λ >= μ !!!"""

        # Number of individuals that have to be generated as initial population.
        num_inputs = 20
        max_generations = 2

        ea_support = EASupport(self.current_space)
        pop_generator = ea_support.generate_initial_population(rand, {'num_inputs': num_inputs})
        ssb = SearchSpaceBounder(self.current_space)

        es = cma.CMAEvolutionStrategy(pop_generator,
                                      sigma,
                                      {
                                          # Population size, AKA lambda, int(popsize) is the number of new solution
                                          # per iteration.
                                          'popsize': num_offspring,
                                          # Random number seed for 'numpy.random'; 'None' and '0' equate to 'time'.
                                          'seed': 0,
                                          # Parents selection parameter, default is popsize // 2, AKA mu.
                                          'CMA_mu': pop_size
                                      })

        for i in range(0, max_generations):
            # Gets list of new solutions (the size of the list is exactly equals to the value of the 'num_offspring'
            # variable). The method returns a list of numpy.ndarray(s), while the evaluate_candidates method expects
            # a simple list.
            candidates = es.ask()
            rearranged_candidates = [np_array.tolist() for np_array in candidates]

            # Moreover, we've to manually apply the search space bounder to each candidate.
            bounded_candidates = [ssb(candidate, {}) for candidate in rearranged_candidates]

            # Now we pass the candidates to the evaluator, fitting them to the model.
            fitnesses = self.evaluate_candidates(bounded_candidates, {})
            es.tell(candidates, fitnesses)

        """Post optimization procedures."""

        # Final population, best unbounded individual, best training fitness.
        final_pop = es.ask()
        best_unbounded_individual = es.best.x
        best_training_fitness = es.best.x

        # Here, again, we're dealing with a ndarray data type that has to be bounded and converted.
        best_individual = ssb(best_unbounded_individual.tolist(), {})

        # 'Reverts' the fitness score.
        if self.maximize:
            best_training_fitness = -best_training_fitness

        # Re-runs the model with the best parameters.
        perf, best_trainer = self.evaluate_candidate(best_individual)

        # We need, also, to set these instance variables to let the Solver access them.
        self.best_trainer = best_trainer
        self.best_para = best_individual

        for ind in final_pop:
            print(str(ind))

        print('Best training fitness = ', best_training_fitness)
        print('\n\n\nDONE.\n\n\n')

        return best_trainer, best_individual
