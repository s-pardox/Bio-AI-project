# Bio-Inspired AI project - UniTN 2021/22

The aim of this project was to apply evolutionary algorithms (EAs) to AutoGL: a class of nature-inspired population-based stochastic search algorithms applied to AutoGL (i.e. Auto Graph Learning), an automatic machine learning (AutoML) toolkit specified for graph datasets and tasks.

EAs are so diverse, as well as their parametrization (that can be summarized in the well known exploration-exploitation trade-off), that researchers may find it difficult to choose which algorithm and parameters should be used.

In addition to technically converging two Python packages (Inspyred and AutoGL), we also pre-parameterized various execution regimes, so that it was possible to compare different algorithms characterized by a higher/lower exploration/exploitation attitude.

### Available algorithms

1. GA: Genetic Algorithm
2. PSO: Particle Swarm Optimization
3. DE: Differential Evolution
4. ES_plus: (μ + λ) Evolution Strategy
5. ES_comma: (μ, λ) Evolution Strategy
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

The program will check automatically if a GPU is available for the task. If not, the CPU will be used to complete the adaptation.

In the case you need to run more than one experiment at time, you can automatize the overall execution by using a simple bash script.

```
#!/bin/bash
python set_ea_hp.py -alg GA -max_eval 450 -pop_size 30 -cr_rate 0.95 -mu_rate 0.05
for i in {1..5}
do
   python run.py -alg GA -wandb_group_name CORA-GCN_GA-Exploitative
done
```

By default, every run produces few output lines on the screen and, at the end, it saves the most relevant statistics on an external CSV file (inside the report/ folder), as well on a remote WandB instance.

### 3. Exploratory data analysis

As third and final step, we developed a module to take care of data analysis, extracting data from WandB with the aim to generate summary aggregated graphs as well individual accuracy/diversity graphs. 
All the plots will be stored into the report/ folder.

```
python3 eda.py
```

More precisely, the execution returns:
    1. a plot consisting of subplots, each corresponding to a specific EA, showing Accuracy vs Diversity;
    2. a comparative plot of the various algorithms, grouped by Explorative-Exploitative-Neutral regimes.

Moreover, a dedicated script produces a parallel coordinates plot.

```
python3 report/parcord.py
```

### WandB Credentials

Our implementation makes massive use of WandB. Edit the template file located in `autogl_wa/settings/wandb_settings.py.template` 
with the desired credentials, as well as the `run.py` and `eda.py` file with the proper run names (and entities).

## Results

TODO: Brief presentation of our findings.