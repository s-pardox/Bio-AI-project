# For example, create a random HPO by yourself
import random

# autogl/module/hpo/base.py
from autogl.module.hpo.base import BaseHPOptimizer

class TestOptimizer(BaseHPOptimizer):
    # Get essential parameters at initialization
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evals = kwargs.get('max_evals', 2)

    # The most important thing you should do is completing optimization function
    def optimize(self, trainer, dataset, time_limit=None, memory_limit=None):
        # 1. Get the search space from trainer.
        # http://mn.cs.tsinghua.edu.cn/autogl/documentation/docfile/tutorial/t_hpo.html#search-space
        space = trainer.hyper_parameter_space + trainer.model.hyper_parameter_space

        # optional: use self._encode_para (in BaseOptimizer) to pretreat the space
        # If you use _encode_para, the NUMERICAL_LIST will be spread to DOUBLE or INTEGER, LOG scaling type will be
        # changed to LINEAR, feasible points in CATEGORICAL will be changed to discrete numbers.
        # You should also use _decode_para to transform the types of parameters back.

        # Ok, we can appreciate this 'facility' during a step-by-step run.
        current_space = self._encode_para(space)

        # 2. Define your function to get the performance.
        def fn(dset, para):

            """
            Copy and paste from node_classifier.py

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

            rearranged_params = {
                'trainer': {                                                # Trainer's parameters
                    'max_epoch': para['max_epoch'],                         # int
                    'early_stopping_round': para['early_stopping_round'],   # int
                    'lr': para['lr'],                                       # float
                    'weight_decay': para['weight_decay']                    # float
                },
                'encoder': {                                                # NN's (encoder) parameters
                    'num_layers': para['num_layers'],                       # int
                    'hidden': para['hidden'],                               # list (number of hidden units)
                    'dropout': para['dropout'],                             # float
                    'act': para['act']                                      # str (activation function)
                },
                'decoder': {}
            }

            # current_trainer = trainer.duplicate_from_hyper_parameter(para)
            # Accordingly with the inner method's behavior, we need to pass a rearranged data structure.

            # The method returns a new instance of the trainer (e.g.: NodeClassificationFullTrainer, solver/classifier/
            # node_classifier.py), also characterized by the parameters we're interested in, as well as by useful
            # methods.
            current_trainer = trainer.duplicate_from_hyper_parameter(rearranged_params)
            current_trainer.train(dset)

            loss, self.is_higher_better = current_trainer.get_valid_score(dset)
            # For convenience, we change the score which is higher better to negative, then we should only minimize
            # the score.
            if self.is_higher_better:
                loss = -loss
            return current_trainer, loss

        # 3. Define the how to get HP suggestions, it should return a parameter dict.
        # # You can use history trials to give new suggestions
        def get_random(history_trials):
            hps = {}
            for para in current_space:

                # Because we use _encode_para function before, we should only deal with DOUBLE, INTEGER and DISCRETE
                if para['type'] == 'DOUBLE' or para['type'] == 'INTEGER':
                    hp = random.random() * (para['maxValue'] - para['minValue']) + para['minValue']
                    if para['type'] == 'INTEGER':
                        hp = round(hp)
                    hps[para['parameterName']] = hp

                elif para['type'] == 'DISCRETE':
                    feasible_points = para['feasiblePoints'].split(',')
                    hps[para['parameterName']] = random.choice(feasible_points)
            return hps

        def get_evolutionary(history_trials):
            hps = {}
            for para in current_space:

                # Because we use _encode_para function before, we should only deal with DOUBLE, INTEGER and DISCRETE
                if para['type'] == 'DOUBLE' or para['type'] == 'INTEGER':
                    hp = random.random() * (para['maxValue'] - para['minValue']) + para['minValue']
                    if para['type'] == 'INTEGER':
                        hp = round(hp)
                    hps[para['parameterName']] = hp

                elif para['type'] == 'DISCRETE':
                    feasible_points = para['feasiblePoints'].split(',')
                    hps[para['parameterName']] = random.choice(feasible_points)
            return hps

        # 4. Run your algorithm. For each turn, get a set of parameters according to history information and evaluate
        # it.
        best_trainer, best_para, best_perf = None, None, None
        self.trials = []

        """
        ORIGINAL SOURCE CODE
        
        for i in range(self.max_evals):
            # in this example, we don't need history trails. Since we pass None to history_trails
            new_hp = get_random(None)
            # optional: if you use _encode_para, use _decode_para as well. para_for_trainer undos all transformation
            # in _encode_para, and turns double parameter to integer if needed. para_for_hpo only turns double
            # parameter to integer.
            para_for_trainer, para_for_hpo = self._decode_para(new_hp)

            # perf == loss
            current_trainer, perf = fn(dataset, para_for_trainer)
        """
        for i in range(self.max_evals):
            # in this example, we don't need history trails. Since we pass None to history_trails
            new_hp = get_random(None)
            # optional: if you use _encode_para, use _decode_para as well. para_for_trainer undos all transformation
            # in _encode_para, and turns double parameter to integer if needed. para_for_hpo only turns double
            # parameter to integer.
            para_for_trainer, para_for_hpo = self._decode_para(new_hp)

            # perf == loss
            current_trainer, perf = fn(dataset, para_for_trainer)

            print('\n__________________________________________________')
            print('Run nr. {}'.format(i))
            print('Performance = {}'.format(perf))
            print('new_hp = \n{}'.format(new_hp))
            print('para_for_trainer = \n{}'.format(para_for_trainer))
            print('para_for_hpo = \n{}'.format(para_for_hpo))

            self.trials.append((para_for_hpo, perf))
            if not best_perf or perf < best_perf:
                best_perf = perf
                best_trainer = current_trainer
                best_para = para_for_trainer

        self.best_trainer = best_trainer
        self.best_para = best_para

        print('\n')

        # 5. Return the best trainer and parameter.
        return best_trainer, best_para