class SearchSpaceMng:
    """
    This class is responsible for modifying the search space so that it is possible, for example, to manipulate the
    number of hidden layers of the NN architecture and, consequently, the length of the genotype.
    """

    HIDDEN_MIN_VALUE = 4
    HIDDEN_MAX_VALUE = 16

    def __init__(self, ss):
        """
        Input parameters:
            ss: AutoGL encoded search space (as originally defined in autogl_ea.settings.search_spaces.py);
        """
        self.ss = ss

    def modify_dv_by_hl(self, orig_dv):
        """
        Input parameters:
            orig_dv: original design variable names list, as defined in the setting file.
        Returns an ordered list containing an updated version of design variables names.
        """
        dv = orig_dv

        num_layers = self.ss[4]['value']
        for i in range(num_layers - 2, 0, -1):
            dv.insert(1, 'hidden_' + str(i))

        return dv

    def modify_ss_by_hl(self, hl):
        """
        Input parameters:
            hl: the number of hidden layers.
        Returns an updated search space, accordingly to the number of specified hidden layers.
        """

        # num_layers
        self.ss['model_hp_space'][0][0]['value'] = hl + 1

        # hidden
        model_ss = self.ss['model_hp_space'][0][1]
        model_ss['minValue'] = [self.HIDDEN_MIN_VALUE for i in range(hl)]
        model_ss['maxValue'] = [self.HIDDEN_MAX_VALUE for i in range(hl)]
        model_ss['length'] = hl
        self.ss['model_hp_space'][0][1] = model_ss

        return self.ss
