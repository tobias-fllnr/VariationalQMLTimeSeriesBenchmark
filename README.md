# VariationalQMLTimeSeriesBenchmark


This repository contains the code to reproduce the results of the paper: 

<cite>Fellner, Tobias, et al. "Quantum vs. classical: A comprehensive benchmark study for predicting time series with variational quantum machine learning." arXiv preprint arXiv:2504.12416 (2025).</cite>

## Installation

To install the required packages, run:

```pip install -r requirements.txt```

## Structure of the repository

- the `.sh` scripts to run the trainings and hyperparameter optimization are in the main directory
- `TimeseriesData` contains notebooks that generate the timeseries data used in the paper. Also plots of the data are generated. Also `nolds_lyapunov_r.ipynb` is used to calculate the Lyapunov times of the data sets.
- `utils` contains all relevant scripts for data preparation, model set up, training and evaluation
- `Training_configurations` contains the different hyperparameter configurations of all models that are trained
- `Results` contains `.csv` files containing the structured training results as well as the results of the hyperparameter optimization
- `Ru_QNN_ansatz` contains the random ansatz generation of the ru-QNN model
- `Plots` contains the plots of the paper as well as the notebooks that were used to create them

## Running the training and evaluation

To run the training of all models and all hyperparameters used in the paper run `start_training.sh`. The default setting is that the code can be executed on any machine. However the optimal way is to execute the training on a slurm cluster. When changing the flag `-slurm "no"` to `-slurm "yes"` the training can efficiently be parallelized on a slurm cluster. 

Since different random initializations are only tested for the best ten ru-QNN ansätze, these models have to be trained after all single initialized ru-QNN models are trained. To find the best ten ansätze for each learning problem and to start the training of these run `optimize_ru_QNN.sh`. 

After all models have been trained run `hyperparameter_optimization.sh` to perform the hyperparameter optimization. This will create the `.csv` files in the folder `Results` with the detailed results and best hyperparameter configurations of each model and for each learning task. These files are subsequently used for creating the plots.
