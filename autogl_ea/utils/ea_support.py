class EASupport:

    def __init__(self, search_space, design_variables):
        self.search_space = search_space
        self.design_variables = design_variables

    def generate_initial_population(self, random, args):
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
        # keep the order of parameters at least, as specified in PARAM_KEYS.
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
