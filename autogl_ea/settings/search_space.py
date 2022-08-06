# Default design variables names (with a default number of hidden layers equals to 1).
DESIGN_VARIABLES = ['hidden_0', 'lr_', 'weight_decay_', 'dropout_', 'act_', 'max_epoch_', 'early_stopping_round_']

"""For GCN model, chromosome is represented as:
    0. H1: hidden_0                [4, 16]

    1. P1: lr                      [1e-2, 5e-2]
    2. P2: weight_decay            [1e-4, 1e-3]
    3. P3: dropout                 [0.2, 0.8]
    4. P4: act                     [0, 3]
    5, P5: max_epoch               [100, 300]
    6. P6: early_stopping_round    [10, 30]
"""
SEARCH_SPACE = {
    'trainer_hp_space': [
        {
            # In Bu et al.'s paper: P1 - continuous param in the [0.01,0,05] range.
            'parameterName': 'lr',
            'type': 'DOUBLE',
            'maxValue': 5e-2,
            'minValue': 1e-2,
            'scalingType': 'LOG',
        },
        {
            # In Bu et al.'s paper: P2 - continuous param in the [0.0001,0.001] range.
            'parameterName': 'weight_decay',
            'type': 'DOUBLE',
            'maxValue': 1e-3,
            'minValue': 1e-4,
            'scalingType': 'LOG',
        },
        {
            # In Bu et al.'s paper: P5 - discrete param in the [100,300] range.
            'parameterName': 'max_epoch',
            'type': 'INTEGER',
            'maxValue': 300,
            'minValue': 100,
            'scalingType': 'LINEAR',
        },
        {
            # In Bu et al.'s paper: P6 - discrete param in the [10,30] range.
            'parameterName': 'early_stopping_round',
            'type': 'INTEGER',
            'maxValue': 30,
            'minValue': 10,
            'scalingType': 'LINEAR',
        }
    ],

    'model_hp_space': [
        [
            # 'encoder'
            {
                # We've temporarily fixed the number of layers to '2' (and consequently, to '1' the number of hidden
                # layers). It can be easily modified using command line arguments.
                'parameterName': 'num_layers',
                'type': 'FIXED',
                'value': 2,
            },
            {
                # In Bu et al.'s paper: H1 - discrete param in the [4,16] range.
                'parameterName': 'hidden',
                'type': 'NUMERICAL_LIST',
                'numericalType': 'INTEGER',
                # Has to be considered as 'max length'.
                'length': 1,
                'minValue': [4],
                'maxValue': [16],

                # Accordingly to Bu et al.'s paper, the values have to be transformed as ln(H1)
                # Does the LOG scale perform a 'ln' transformation? Yes, it does.
                'scalingType': 'LOG',

                # By expliciting 'cutPara' we force HPO to cut the list to a certain length which is dependent on
                # 'num_layers' param.
                'cutPara': ('num_layers',),
                # As general rule:
                #   len(hidden) = num_layers - 1
                'cutFunc': lambda x: x[0] - 1,
            },
            {
                # In Bu et al.'s paper: P3 - continuous param in the [0.2,0.8] range.
                'parameterName': 'dropout',
                'type': 'DOUBLE',
                'maxValue': 0.8,
                'minValue': 0.2,
                'scalingType': 'LINEAR',
            },
            {
                # In Bu et al.'s paper: P4 - discrete categorical param.
                'parameterName': 'act',
                'type': 'CATEGORICAL',
                'feasiblePoints': ['leaky_relu', 'relu', 'elu', 'tanh'],
            }
        ]
        # We don't have any 'decoder' counter-part parameters.
    ]
}
