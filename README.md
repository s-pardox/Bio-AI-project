# Bio-Inspired AI project - UniTN 2021/22

## Introduction

The aim of this project was to apply evolutionary algorithms (EAs), as a class of nature-inspired 
population-based stochastic search algorithms applied, to Auto Graph Learning 
([AutoGL](https://github.com/THUMNLab/AutoGL)), an automatic machine learning toolkit specified for Graph Neural Network (GNN).

EAs are so diverse, as well as their parametrization (synthesizable in the well known exploration-exploitation 
trade-off), that researchers may find it difficult to choose which algorithm and parameters should be used.

In addition to technically converging two Python packages ([Inspyred](https://github.com/aarongarrett/inspyred) and 
[AutoGL](https://github.com/THUMNLab/AutoGL)), we also pre-parameterized various execution regimes with the aim to 
compare different algorithms, characterized by a higher/lower exploration/exploitation attitude.

The experimental results showed that EAs could be an effective alternative to the hyperparameter optimization (HPO) of 
GNN models.

**Please refer to the [final report](https://github.com/s-pardox/Bio-AI-project/blob/main/report/Bio-AI_report.pdf) for 
a more comprehensive discussion.**

### Available algorithms

1. GA: Genetic Algorithm
2. PSO: Particle Swarm Optimization
3. DE: Differential Evolution
4. ES_plus: (μ + λ) Evolution Strategy
5. ES_comma: (μ, λ) Evolution Strategy
6. CMA-ES: Covariance Matrix Adaptation - Evolution Strategy

## Virtual environment setup and requirements

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

Otherwise, you can set up your own execution environment with Anaconda.

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

The program will check automatically if a GPU is available for the task. If not, the CPU will be used to complete the 
adaptation.

In the case you need to run more than one experiment at time, you can automatize the overall execution by using a simple 
bash script.

```
#!/bin/bash
python set_ea_hp.py -alg GA -max_eval 450 -pop_size 30 -cr_rate 0.95 -mu_rate 0.05
for i in {1..5}
do
   python run.py -alg GA -wandb_group_name CORA-GCN_GA-Exploitative
done
```

By default, every run produces few output lines on the screen and, at the end, it saves the most relevant statistics on 
an external CSV file (in the `report/` folder), as well on a remote WandB instance.

### 3. Exploratory data analysis

As third and final step, we developed a module to take care of data analysis. By extracting data from WandB, it 
generates summary aggregated graphs, as well individual accuracy/diversity graphs. 
All the plots will be stored into the `report/` folder.

```
python3 eda.py
```

More precisely, the execution returns:
1. a plot consisting of subplots, each corresponding to a specific EA, showing Accuracy vs Diversity;
2. a comparative plot of the various algorithms, grouped by Explorative-Exploitative-Neutral regimes.

Moreover, a dedicated script produces a parallel coordinates plot, showing a comparison of different 
hyperparameters set (by using the best candidate solution found for each algorithm).

```
python3 report/parcord.py
```

### WandB Credentials

Our implementation makes massive use of WandB. Edit the template file located in `autogl_wa/settings/wandb_settings.py.template` 
with the desired credentials, as well as the `run.py` and `eda.py` file with the proper run names (and entities).

## Results

### 1. Accuracy vs Diversity

The most interesting behaviour regards the CMA-ES algorithm: while the accuracy function increases over time, 
the diversity measure shows a similar trend as well. It tells us that better results, in terms of accuracy, would
probably have been obtained with more generations.

### 2. Architecture comparison

The only clear pattern that emerges regards the *Early Stopping* parameter. In both dataset used (Cora and Citeseer), 
its optimal value ranges between 20 and 40, avoiding high values.

Although the results are not so significant, we can summarize this analysis as follows:
different sets of GCN hyperparameters can lead to almost equal good results.

### 3. Which algorithmic approach is better?

Referring to all the runs performed, it doesn't emerge a clear "winner" in terms of test accuracy.
The difference of the worst-best algorithms is around 2%.

A more Explorative algorithmic regime seems to be preferable on the Cora dataset, while an Exploitative approach should 
better fit Citeseer.

**Please refer to the [final report](https://github.com/s-pardox/Bio-AI-project/blob/main/report/Bio-AI_report.pdf) for a more 
comprehensive discussion.**
