# For example, create a random HPO by yourself
import random

import inspyred
import time

# autogl/module/hpo/base.py
from autogl.module.hpo.base import BaseHPOptimizer
from inspyred.ec import Bounder, DiscreteBounder


class SearchSpaceBounder:
    """This class is inspired by inspyred.ec.Bounder/DiscreteBounder but has been completely rewritten in order to
    push the AutoGL search space into the evolutionary process.
    The original classes define a basic bounding function for numeric lists of discrete/continuous values.
    """

    # Parameters that, during the evolutionary cycle, have to correctly bounded before to be passed to the trainer.
    to_bound = ['dropout_', 'act_']

    def __init__(self, param_keys, current_space):
        self.param_keys = param_keys
        self.current_space = current_space

    def __call__(self, candidate, args):
        """
        Input parameters:
            candidate: a list that contains, in each position, a value representing a specific gene (the index of each
                one is exactly the same of the param_keys list)
            args: dictionary that contains Inspyred's evolutionary parameters (max_generations, num_selected, etc.)
        """
        for para in self.current_space:
            if para['parameterName'] in self.to_bound:
                i = self.param_keys.index(para['parameterName'])

                # Because we use _encode_para function before, we should only deal with DOUBLE, INTEGER and DISCRETE
                if para['type'] == 'DOUBLE' or para['type'] == 'INTEGER':
                    candidate[i] = max(min(candidate[i], para['maxValue']), para['minValue'])

                    """
                    Is it really necessary?
                    if para['type'] == 'INTEGER':
                        candidate[i] = round(candidate[i])
                    """

                elif para['type'] == 'DISCRETE':
                    feasible_points = para['feasiblePoints'].split(',')
                    closest = lambda target: min(feasible_points, key=lambda x: abs(int(x) - target))
                    candidate[i] = int(closest(candidate[i]))

        return candidate


class InspyredOptimizer(BaseHPOptimizer):
    # Get essential parameters at initialization.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get('max_evals', 2)
        # Defaul value.
        self.alg = kwargs.get('alg', 'GA')

        """For GCN model, chromosome is represented as:
            0. H1: hidden_0                [4,16]
            
            1. P1: lr                      [1e-2, 5e-2]
            2. P2: weight_decay            [1e-4, 1e-3]
            3. P3: dropout                 [0.2, 0.8]
            4. P4: act                     [0,3]
            5, P5: max_epoch               [100,300]
            6. P6: early_stopping_round    [10, 30]
        """
        self.param_keys = ['hidden_0', 'lr_', 'weight_decay_', 'dropout_', 'act_', 'max_epoch_',
                           'early_stopping_round_']

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        """This method is automatically invoked by AutoGL; has to be considered the entry point of the optimization
        process.
        """
        # Get the search space from trainer.
        # http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_hpo.html#search-space
        space = trainer.hyper_parameter_space + trainer.model.hyper_parameter_space

        # Optional: use self._encode_para (in BaseOptimizer) to pretreat the space.
        # If you use _encode_para, the NUMERICAL_LIST will be spread to DOUBLE or INTEGER, LOG scaling type will be
        # changed to LINEAR, feasible points in CATEGORICAL will be changed to discrete numbers.
        # You should also use _decode_para to transform the types of parameters back.

        # Ok, we can appreciate this 'facility' during a step-by-step debugging run.
        current_space = self._encode_para(space)

        def generate_initial_population(random, args):
            """Initializes the initial population.

            This method initializes a single individual belonging to the initial population. It simply randomly
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

            # In the case the pop_size parameter wasn't specified in ga.evolve() method.
            size = args.get('num_inputs', 10)

            # For each individual, due to the fact we cannot keep a key-value parameter pair, we'd like to
            # keep the order of parameters at least, as specified in 'param_keys'.
            individual = []
            for param_key in self.param_keys:

                for para in current_space:
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

        def __rearrange_params(para):
            """Copy and paste from node_classifier.py:
            hp: ``dict``.
                The hyperparameter used in the new instance. Should contain 3 keys "trainer", "encoder"
                "decoder", with corresponding hyperparameters as values.

            So, we have to rearrange the dictionary to avoid this error:
                File "[...]/autogl/module/train/node_classification_full.py", line 524, in duplicate_from_hyper_parameter
                lr=hp["lr"],
                KeyError: 'lr'

            An example with real values:
            rearranged_params = {
                'trainer': {
                    'max_epoch': 101,
                    'early_stopping_round': 26,
                    'lr': 0.00011459309016222376,
                    'weight_decay': 0.00
                },
                'encoder': {
                    'num_layers': 2,
                    'hidden': [32],
                    'dropout': 0.5484903922544934,
                    'act': 'elu'
                },
                'decoder': {}
            }
            """

            return {
                'trainer': {  # Trainer's parameters
                    'max_epoch': para['max_epoch'],  # int
                    'early_stopping_round': para['early_stopping_round'],  # int
                    'lr': para['lr'],  # float
                    'weight_decay': para['weight_decay']  # float
                },
                'encoder': {  # NN's (encoder) parameters
                    'num_layers': para['num_layers'],  # int
                    'hidden': para['hidden'],  # list (number of hidden units)
                    'dropout': para['dropout'],  # float
                    'act': para['act']  # str (activation function)
                },
                'decoder': {}
            }

        def __fit(individual):
            """Receives a list of hyperparameters (a genetic representation of an individual) to fit the model with.

            Returns:
                1. the performance in the [0,1] range;
                2. an instance of the trainer.
            """

            """Because the candidate solution is composed by unnamed genes (simply a list of values), we have first
            to reassign a key to each one, before to evaluate it with the trainer.
            """
            named_individual = {}
            for i in range(0, len(individual) - 1):
                if self.param_keys[i] == 'act_':
                    # TODO.
                    # Reverts the value from float to str, as needed by _decode_para method.
                    named_individual[self.param_keys[i]] = str(individual[i])
                else:
                    named_individual[self.param_keys[i]] = individual[i]

            # Decode.
            para_for_trainer, para_for_hpo = self._decode_para(named_individual)
            # Rearrange.
            rearranged_params = __rearrange_params(para_for_trainer)

            # The method returns a new instance of the trainer
            # (e.g.: NodeClassificationFullTrainer, solver/classifier/node_classifier.py),
            # also characterized by the parameters we're interested in, as well as by useful methods.
            current_trainer = trainer.duplicate_from_hyper_parameter(rearranged_params)
            current_trainer.train(dataset)

            loss, self.is_higher_better = current_trainer.get_valid_score(dataset)

            """                
            # For convenience, we change the score which is higher better to negative, then we should only minimize
            # the score.
            if self.is_higher_better:
                loss = -loss
            return current_trainer, loss
            """

            # Performance.
            perf = loss
            return perf, current_trainer

        def evaluate_candidates(candidates, args):
            """Evaluates a candidate by running a training cycle and getting the training performance.

            We've taken, as example, this function:
                def evaluate_rastrigin(candidates, args):
                    fitness = []
                    for cs in candidates:
                        fit = 10 * len(cs) + sum([((x - 1) ** 2 - 10 * cos(2 * pi * (x - 1))) for x in cs])
                        fitness.append(fit)
                    return fitness
            (Reference: https://pythonhosted.org/inspyred/_downloads/rastrigin.py)

            So, basically, we've to return a list of values/fitness, one for each candidate solution.
            """

            fitness = []
            for individual in candidates:
                # Evaluates the model using the evolved parameters.
                perf, _ = __fit(individual)
                fitness.append(perf)

            return fitness

        def GA():
            """Classic genetic algorithm"""

            rand = random.Random()
            rand.seed(int(time.time()))
            ga = inspyred.ec.GA(rand)
            ga.observer = inspyred.ec.observers.stats_observer
            ga.terminator = inspyred.ec.terminators.evaluation_termination

            return ga.evolve(evaluator=evaluate_candidates,
                             #
                             generator=generate_initial_population,
                             # Number of generations = max_evaluations / pop_size.
                             max_evaluations=30,
                             #
                             num_elites=5,
                             # Population size.
                             pop_size=30,
                             # Number of individuals that have to be generated as initial population.
                             num_inputs=10,
                             #
                             bounder=SearchSpaceBounder(self.param_keys, current_space))

        def PSO():
            """Particle Swarm Optimization"""

            # Main outer training cycle controlled by Inspyred
            rand = random.Random()
            rand.seed(int(time.time()))
            ea = inspyred.swarm.PSO(rand)
            ea.topology = inspyred.swarm.topologies.ring_topology
            ea.terminator = inspyred.ec.terminators.evaluation_termination

            return ea.evolve(evaluator=evaluate_candidates,
                             generator=generate_initial_population,
                             pop_size=25,
                             max_evaluations=100,
                             neighborhood_size=5,
                             bounder=SearchSpaceBounder(self.param_keys, current_space))

        def DEA():
            """Differential Evolution"""

            rand = random.Random()
            rand.seed(int(time.time()))
            ea = inspyred.ec.DEA(rand)
            ea.terminator = inspyred.ec.terminators.evaluation_termination
            return ea.evolve(generator=generate_initial_population,
                             evaluator=evaluate_candidates,
                             pop_size=25,
                             bounder=SearchSpaceBounder(self.param_keys, current_space),
                             max_generations=30)

        current_space = self._encode_para(trainer.hyper_parameter_space + trainer.model.hyper_parameter_space)

        def ES_1():
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

            rand = random.Random()
            rand.seed(int(time.time()))
            ea = inspyred.ec.ES(rand)
            ea.terminator = [inspyred.ec.terminators.evaluation_termination,
                             inspyred.ec.terminators.diversity_termination]
            return ea.evolve(generator=generate_initial_population,
                             evaluator=evaluate_candidates,
                             # mu parameter, as defined in Bu et al.'s paper
                             pop_size=100,
                             bounder=SearchSpaceBounder(self.param_keys, current_space),
                             max_generations=2)

        current_space = self._encode_para(trainer.hyper_parameter_space + trainer.model.hyper_parameter_space)

        if self.alg == 'GA':
            print('\nRunning GA...')
            final_pop = GA()

        elif self.alg == 'PSO':
            print('\nRunning PSO...')
            final_pop = PSO()

        elif self.alg == 'DEA':
            print('\nRunning DEA...')
            final_pop = DEA()

        elif self.alg == 'ES_1':
            print('\nRunning (μ/ρ,λ)-ES Evolution Strategy:')
            final_pop = ES_1()

        # Instance of Individual class.
        best_individual_obj = max(final_pop)
        # Extracts the best individual... (list)
        best_individual = best_individual_obj.candidate
        # ...and its training fitness.
        best_fitness = best_individual_obj.fitness

        # Re-runs the model with the best parameters.
        perf, best_trainer = __fit(best_individual)

        # We need, also, to set these instance variables to let the Solver access them.
        self.best_trainer = best_trainer
        self.best_para = best_individual

        for ind in final_pop:
            print(str(ind))

        print('\n\n\nDONE.\n\n\n')

        return best_trainer, best_individual
