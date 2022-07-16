"""
UniTN Bio Inspired Artificial Intelligent final project - 2021/2022
Authors: Mattia Florio, Stefano Pardini

This module is the entry point of the program, it is responsible for parsing the command line parameters, launching
functions from autogl_ea.app package.
"""

import os
import argparse
import sys

import autogl_ea.app as app


def main():
    """
    TODO.
    Command line arguments parser, WanDB initializer, launcher.
    """

    # TODO.
    # Available options for 'alg': 'GA', 'PSO', 'DEA', 'ES', 'CMA-ES'
    # Here we should choose the dataset, the graph model, etc.
    app.launch(alg='PSO', dataset='cora', graph_model=['gcn'])


if __name__ == '__main__':
    main()
