import yaml

# autogl/module/hpo/base.py
from autogl.module.hpo.base import BaseHPOptimizer

import autogl_ea.settings.search_space as ss
from autogl_ea.utils.search_space import SearchSpaceMng
from autogl_ea.utils import EASupport
from autogl_ea.settings import config as cfg


class HPOptimizer(BaseHPOptimizer):
    # Get essential parameters at initialization.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_evals = kwargs.get('max_evals', 2)

        # The following instance variables are going to be set in the optimize method.
        self.trainer = None
        self.dataset = None
        self.space = None
        self.current_space = None
        self.design_variables = None

        # Do we need to invert the optimization logic? (Are we looking for minimum or maximum?)
        self.maximize = False

        # Calculated in post_Inspyred_optimization method, after the evolutionary process ending.
        self.best_trainer = None
        self.best_para = None
        self.diversity = None

    def get_config(self):
        """This method gets the hyperparameters of the EA from the yaml file
        """

        config = dict()

        with open(cfg.EA_HP_PATH, 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            config = params.copy()

        return config

    def rearrange_params(self, para):
        """Copied and pasted from node_classifier.py:
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

    def gen_named_individual(self, individual):
        """Because the candidate solution is composed by unnamed genes (simply a list of values), we have first
        to reassign a key to each one, before to evaluate it with the trainer.
        """
        named_individual = {}
        # Starting from the 2nd generation of comma/plus ES, to every individual are added the mutations rates. So,
        # we cannot directly use the individual length: in those cases is exactly doubled.
        # for i in range(0, len(individual)):
        for i in range(0, len(self.design_variables)):
            if self.design_variables[i] == 'act_':
                # TODO.
                # Reverts the value from float to str, as needed by _decode_para method.
                named_individual[self.design_variables[i]] = str(individual[i])
            else:
                named_individual[self.design_variables[i]] = individual[i]

        return named_individual

    def fit(self, individual):
        """Receives a list of hyperparameters (a genetic representation of an individual) to fit the model with.

        Returns:
            1. the performance in the [0,1] range;
            2. an instance of the trainer.
        """

        # Assigns a key to every gene.
        named_individual = self.gen_named_individual(individual)

        # Decode.
        para_for_trainer, para_for_hpo = self._decode_para(named_individual)
        # Rearrange.
        rearranged_params = self.rearrange_params(para_for_trainer)

        # The method returns a new instance of the trainer
        # (e.g.: NodeClassificationFullTrainer, solver/classifier/node_classifier.py),
        # also characterized by the parameters we're interested in, as well as by useful methods.
        current_trainer = self.trainer.duplicate_from_hyper_parameter(rearranged_params)
        current_trainer.train(self.dataset)

        loss, self.is_higher_better = current_trainer.get_valid_score(self.dataset)

        """
        # For convenience, we change the score which is higher better to negative, then we should only minimize
        # the score.
        if self.is_higher_better:
            loss = -loss
        return current_trainer, loss
        """

        # This is useful in particular with CMA-ES, because it searches for the minimum, by default.
        if self.maximize:
            loss = -loss

        # Performance.
        perf = loss
        return perf, current_trainer

    def evaluate_candidates(self, candidates, args):
        """Evaluates a list of candidate by running a training cycle and getting the training performance.

        We've taken, as example, this function:
            def evaluate_rastrigin(candidates, args):
                fitness = []
                for cs in candidates:
                    fit = 10 * len(cs) + sum([((x - 1) ** 2 - 10 * cos(2 * pi * (x - 1))) for x in cs])
                    fitness.append(fit)
                return fitness
        (Reference: https://pythonhosted.org/inspyred/_downloads/rastrigin.py)

        So, basically, we've to return a list of values/fitness, one for each candidate solution.
        Intuitively, the method expects a list of lists (candidates) as input.
        """

        fitness = []
        for individual in candidates:
            # Evaluates the model using the evolved parameters.
            perf, _ = self.fit(individual)
            fitness.append(perf)

        return fitness

    def evaluate_candidate(self, individual):
        """Evaluates a single candidate by running a training cycle and getting the training performance."""
        return self.fit(individual)

    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        """This method is automatically invoked by AutoGL; has to be considered the entry point of the optimization
        process.
        """

        self.trainer = trainer
        self.dataset = dataset

        # Get the search space from trainer (it has been already initialized with our specifications).
        # http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_hpo.html#search-space
        self.space = trainer.hyper_parameter_space + trainer.model.hyper_parameter_space

        # Optional: use self._encode_para (in BaseOptimizer) to pretreat the space.
        # If you use _encode_para, the NUMERICAL_LIST will be spread to DOUBLE or INTEGER, LOG scaling type will be
        # changed to LINEAR, feasible points in CATEGORICAL will be changed to discrete numbers.
        # You should also use _decode_para to transform the types of parameters back.

        # _encode_para method is inherited from BaseHPOptimizer super class.
        self.current_space = super()._encode_para(self.space)

        # Gets the design variables names, accordingly to the actual search space architecture and, in particular,
        # to the number of hidden layers.
        ss_mng = SearchSpaceMng(self.space)
        self.design_variables = ss_mng.modify_dv_by_hl(ss.DESIGN_VARIABLES)

        """
            Inherit this method and enter the optimization logic of the genetic algorithm here...
        """

    def post_Inspyred_optimization(self, final_pop):
        """Common post optimization procedures to all Inspyred's algorithms."""

        # Instance of Individual class.
        best_individual_obj = max(final_pop)
        # Extracts the best individual... (list)
        best_individual = best_individual_obj.candidate
        # ...and its training fitness.
        best_fitness = best_individual_obj.fitness

        # Re-runs the model with the best parameters.
        perf, best_trainer = self.fit(best_individual)

        # We need, also, to set these instance variables to let the Solver (and app.py, trough solver.hpo_model) access
        # them.
        self.best_trainer = best_trainer
        self.best_para = best_individual
        # Diversity.
        ea_support = EASupport(self.current_space, self.design_variables)
        self.diversity = ea_support.get_diversity(final_pop)

        print('\nFinal population:\n')
        for ind in final_pop:
            print(str(ind))

        print('\nBest training accuracy: {:.4f}'.format(best_fitness))
        print('\n\n\nDONE.\n\n\n')

        return best_trainer, best_individual
