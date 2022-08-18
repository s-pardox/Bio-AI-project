import wandb
import inspyred
import itertools
import math
import statistics


class EASupport:

    def __init__(self, search_space=None, design_variables=None):
        # These instances variables are only used by generate_initial_population method.
        self.search_space = search_space
        self.design_variables = design_variables

    def generate_initial_population(self, random, args):
        """Initializes the initial population.

        This method initializes/generates a single individual belonging to the initial population. It simply randomly
        generate bounded parameters (i.e. a bounded search space), following the rules associated with each of them.

        We need both formal parameters (random, args), because the function is invoked as:
            cs = generator(random=self._random, args=self._kwargs)
        (Reference: inspyred/ec/ec.py, line 430)

        This is an example generation function
            def generate_rastrigin(random, args):
                size = args.get('num_inputs', 10)
                return [random.uniform(-5.12, 5.12) for i in range(size)]
        (Reference: https://pythonhosted.org/inspyred/_downloads/rastrigin.py)

        Remember that:
            "In GCN, the number of layers in the convolution structure was fixed, and only the number of units in
            the hidden layer (H1) was adjusted. It is because when the number of layers is above two, the effect is
            not greatly improved, and when the number of layers is too high, the training effect is significantly
            reduced."
        (Reference: Bu et al.)

        For that reason, we have temporary fixed to '1' the number of hidden layers, letting evolve the number of
        hidden units for that single layer (H1).
        """

        # For each individual, due to the fact we cannot keep a key-value parameter pair, we'd like to
        # keep the order of parameters at least, as specified in DESIGN_VARIABLES.
        individual = []
        for param_key in self.design_variables:

            for para in self.search_space:
                if para['parameterName'] == param_key:

                    # Because we use _encode_para function before, we should only deal with DOUBLE, INTEGER and
                    # DISCRETE
                    if para['type'] == 'DOUBLE' or para['type'] == 'INTEGER':
                        hp = random.random() * (para['maxValue'] - para['minValue']) + para['minValue']
                        if para['type'] == 'INTEGER':
                            hp = round(hp)
                        individual.append(hp)

                    elif para['type'] == 'DISCRETE':
                        feasible_points = para['feasiblePoints'].split(',')
                        """
                            The int cast should be enough to prevent the following error:
                            [...]/swarm.py", line 100, in _swarm_variator
                                value = (xi + inertia * (xi - xpi) + 
                            TypeError: unsupported operand type(s) for -: 'str' and 'str'
    
                            Then, we'll have to cast the type into a string.
                        """
                        # TODO.
                        individual.append(int(random.choice(feasible_points)))
                        # individual.append(random.choice(feasible_points))

                    break

        return individual

    def observer(self, population, num_generations, num_evaluations, args):
        """This method (and the initial comments) has been taken from inspyred.ec.observers.py module.
        We just add the WandB logging at the end.

        Print the statistics of the evolutionary computation to the screen.

        This function displays the statistics of the evolutionary computation
        to the screen. The output includes the generation number, the current
        number of evaluations, the maximum fitness, the minimum fitness,
        the average fitness, and the standard deviation.

        .. note::

           This function makes use of the ``inspyred.ec.analysis.fitness_statistics``
           function, so it is subject to the same requirements.

        .. Arguments:
           population -- the population of Individuals
           num_generations -- the number of elapsed generations
           num_evaluations -- the number of candidate solution evaluations
           args -- a dictionary of keyword arguments

        """
        stats = inspyred.ec.analysis.fitness_statistics(population)
        worst_fit = '{0:>10}'.format(stats['worst'])[:10]
        best_fit = '{0:>10}'.format(stats['best'])[:10]
        avg_fit = '{0:>10}'.format(stats['mean'])[:10]
        med_fit = '{0:>10}'.format(stats['median'])[:10]
        std_fit = '{0:>10}'.format(stats['std'])[:10]

        print('Generation Evaluation      Worst       Best     Median    Average    Std Dev')
        print('---------- ---------- ---------- ---------- ---------- ---------- ----------')
        print('{0:>10} {1:>10} {2:>10} {3:>10} {4:>10} {5:>10} {6:>10}\n'.format(num_generations,
                                                                                 num_evaluations,
                                                                                 worst_fit,
                                                                                 best_fit,
                                                                                 med_fit,
                                                                                 avg_fit,
                                                                                 std_fit))

        diversity = self.get_diversity(population)

        # Logs the statistics on WandB.
        wandb.log({
            'worst_fit': float(worst_fit),
            'best_fit': float(best_fit),
            'med_fit': float(med_fit),
            'avg_fit': float(avg_fit),
            'std_fit': float(std_fit),

            'min_div': diversity['min'],
            'max_div': diversity['max'],
            'med_div': diversity['med'],
            'avg_div': diversity['avg'],
            'std_div': diversity['std']
        })

    def get_diversity(self, population):
        """This method has been inspyred by inspyred.ec.terminators.py module.
        It calculates the Euclidean distance between every pair of individuals in the population and returns some
        basic statistics.
        """
        ind_length = len(self.design_variables)

        cart_prod = itertools.product(population, population)
        distance = []
        for (p, q) in cart_prod:
            d = 0
            # EAs add at the end of every individual a number of mutation rates equals to the number of genes.
            # For that reason we've to truncate the list.
            for x, y in zip(p.candidate[0:ind_length], q.candidate[0:ind_length]):
                d += (x - y) ** 2
            distance.append(math.sqrt(d))

        # Removes all those values result of self comparisons (matrix diagonal).
        distance = list(filter(lambda a: a != 0.0, distance))

        return {'min': min(distance), 'max': max(distance), 'med': statistics.median(distance),
                'avg': statistics.mean(distance), 'std': statistics.stdev(distance)}
