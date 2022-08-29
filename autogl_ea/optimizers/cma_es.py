import random

import cma
import time

from autogl_ea.optimizers import HPOptimizer
from autogl_ea.utils import EASupport
from autogl_ea.utils import SearchSpaceBounder

from inspyred.ec.ec import Individual


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

        config = self.get_config()

        """CMA-ES initialization parameters"""

        # Initial standard deviation.
        sigma = 0.5
        # Random number seed for 'numpy.random'; 'None' and '0' equate to 'time'.
        seed = 0

        # Take a look at the default options ('cma_default_options_') in cma/evolution_strategy.py.
        # We've 'popsize' parameter instead of 'pop_size'.
        # Population size, AKA lambda, int(popsize) is the number of new solution per iteration (initial population
        # and/or the number of offspring that will survive at every generation).
        pop_size = config['pop_size']['value']
        # CMA_mu: parents selection parameter, default is popsize // 2'
        # We can use the default parameter.
        # Very important constraint: λ >= μ. In terms of CMA-ES parameters, the constraint is:
        # popsize (λ) >= CMA_mu (μ)

        num_gen = config['max_gen']['value']

        ea_support = EASupport(self.current_space, self.design_variables)
        pop_generator = ea_support.generate_initial_population(rand, {})
        ssb = SearchSpaceBounder(self.current_space)

        es = cma.CMAEvolutionStrategy(pop_generator,
                                      sigma,
                                      {
                                          'popsize': pop_size,
                                          'seed': seed
                                      })

        def convert_to_inspyred_pop(cma_es_candidates):
            """In order to be able to use the Inspyred observer method, we have to map the candidates into an Inspyred
            Individual list of objects."""
            inspyred_pop = []
            for j, candidate in enumerate(cma_es_candidates):
                individual = Individual(candidate=candidate.tolist())
                individual.fitness = fitnesses[j]
                inspyred_pop.append(individual)
            return inspyred_pop

        """Evolutionary algorithm main cycle"""
        for i in range(0, num_gen):
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

            """Observer"""
            population = convert_to_inspyred_pop(candidates)
            ea_support.observer(population, num_generations=i + 1, num_evaluations=(i + 1) * pop_size, args=None)

        """Post optimization procedures."""

        # Final population, best unbounded individual, best training fitness.
        final_pop = es.ask()
        best_unbounded_individual = es.best.x
        best_training_fitness = es.best.f

        # Here, again, we're dealing with a ndarray data type that has to be bounded and converted.
        best_individual = ssb(best_unbounded_individual.tolist(), {})

        # 'Reverts' the fitness score.
        if self.maximize:
            best_training_fitness = -best_training_fitness

        # Re-runs the model with the best parameters.
        perf, best_trainer = self.evaluate_candidate(best_individual)

        # We need, also, to set these instance variables to let the Solver (and app.py, trough solver.hpo_model) access
        # them.
        self.best_trainer = best_trainer
        self.best_para = best_individual
        # Diversity.
        self.diversity = ea_support.get_diversity(convert_to_inspyred_pop(final_pop))

        print('\nFinal population:\n')
        for ind in final_pop:
            print(str(ind))

        print('\nBest training accuracy: {:.4f}'.format(best_training_fitness))
        print('\n\n\nDONE.\n\n\n')

        return best_trainer, best_individual
