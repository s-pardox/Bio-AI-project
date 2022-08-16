from autogl_ea.settings import search_space as ss


class SearchSpaceBounder:
    """
    This class is inspired by inspyred.ec.Bounder/DiscreteBounder but has been completely rewritten in order to
    push the AutoGL search space into the evolutionary process.
    The original classes define a basic bounding function for numeric lists of discrete/continuous values.

    The Bounder is indispensable to prevent exceptions like this (eventually raised during the fitting moment):
        File "[...]/base.py", line 228, in _decode_para_convert
            externel_para[name] = self._category_map[name][int(para[name])]
        ValueError: invalid literal for int() with base 10: '2.553333004298308'

    It operates at end of the evolution process, after generating a candidate, before fitting it.
    """

    # Parameters that, during the evolutionary process, have to correctly bounded before to be passed to the trainer.
    TO_BOUND = ['dropout_', 'act_']

    def __init__(self, current_space):
        """
        Input parameters:
            current_space: AutoGL encoded search space (as originally defined in autogl_ea.settings.search_spaces.py).
        """
        self.current_space = current_space

    def __call__(self, candidate, args):
        """
        Input parameters:
            candidate: a list that contains, in each position, a value representing a specific gene (the index of each
                one is exactly the same of the DESIGN_VARIABLES list);
            args: dictionary that contains Inspyred's evolutionary parameters (max_generations, num_selected, etc.).
        """
        for para in self.current_space:
            if para['parameterName'] in self.TO_BOUND:
                i = ss.DESIGN_VARIABLES.index(para['parameterName'])

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
