# Bio-Inspired AI project - UniTN 2021/22

TODO: Introduction, brief description of the project.

### Available algorithms

1. GA: Genetic Algorithm
2. PSO: Particle Swarm Optimization,
3. DE: Differential Evolution,
4. ES_plus: (μ + λ) Evolution Strategy,
5. ES_comma: (μ, λ) Evolution Strategy,
6. CMA-ES: Covariance Matrix Adaptation - Evolution Strategy

### Virtual environment setup and requirements

In order to execute the project after the cloning, it is suggested to create a virtual environment and install the 
required modules.  
To create a virtual environment in a Unix-like operating system, type in a terminal:

```
# Use python or python3 (depending on the recognised command)
python3 -m venv ./venv
```

Then, activate your virtual environment:

```
source venv/bin/activate
```

Finally, install all the required dependencies:

```
pip install -r requirements.txt
```

The authors successfully used Python versions 3.9.4 and 3.9.12.

Otherwise, you can use Anaconda.

## Usage

### 1. Hyperparameters setting

First of all it ìs necessary to set the hyperparameters of the evolutionary algorithm that you're going to run. 

```
python3 set_ea_hp.py
```

The script produces the output file `config-defaults.yaml` containing the list with the chosen parameters. They will be 
eventually used by WandB as configuration values, helping to perform a-posteriori analysis.

#### Parameters

* `-alg`: Options: GA, PSO, DE, ES_plus, ES_comma, CMA-ES; default=GA
* `-max_eval`: Set the max number of evaluations; default=None
* `-max_gen`: Set the max number of generations; default=None
* `-pop_size`: Set the population size; default=None
* `-cr_rate`: Set the crossover rate; default=None
* `-mu_rate`: Set the mutation rate; default=None
* `-inertia`: Set the Inertia velocity (only for PSO); default=None
* `-cognitive`: Set the Cognitive velocity (only for PSO); default=None
* `-social`: Set the Social velocity (only for PSO); default=None

### 2. Execution of the experiments

Secondarily, the main program have to be launched.

```
python3 run.py
```

This module is the main entry point of the program: it is responsible for parsing the command line parameters, launching
functions from autogl_ea.app package, activating the evolutionary process for the chosen algorithm.

#### Parameters

* `-alg`: Options: GA, PSO, DE, ES_plus, ES_comma, CMA-ES; default=GA
* `-dataset`: Options: cora, proteins; default=cora
* `-graph_model`: Options: gcn, gat; default=gcn
* `-hl`: The number of hidden layers to be used; default=1
* `-problem`: Classification options: node, graph; default=node
* `-wandb`: Log results on WandB; default=False
* `-wandb_group_name`: 'WandB group name; default=Final Experiments
* `-runs`: Number of executions to be performed for the selected algorithm; default=5

The program will check automatically if a GPU is available for the task. If not, the CPU will be used to complete the adaptation.

### 3. Analysis: generation of graphs and statistics

TODO.

### WandB Credentials

To store run results in a WandB instance, edit the template file stored in `autogl_wa/settings/wandb_settings.py.template` 
with the desired credentials, as well as the `run.py` file with the proper run names (and entities).

## Results

TODO: Brief presentation of our findings.
