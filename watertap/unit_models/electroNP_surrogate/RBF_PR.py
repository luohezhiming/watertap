from idaes.core.surrogate.pysmo_surrogate import (
    PysmoPolyTrainer,
    PysmoKrigingTrainer,
    PysmoRBFTrainer,
    PysmoSurrogate,
)
import os
import numpy as np
import pandas as pd

# Import IDAES libraries
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.pysmo_surrogate import PysmoPolyTrainer, PysmoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
    surrogate_parity,
    surrogate_residual,
)
from idaes.core.surrogate.metrics import compute_fit_metrics

# Load dataset from a csv file
# xy_data = pd.read_csv("PR_DOE_unique_data_restructure_v2.csv", skiprows=1, header=None)
xy_data = pd.read_csv("PR_DOE_unique_data_restructure_v2.csv")
# xy_data_total = pd.read_csv(
#     "PR_complete_DOE_results_restructure_v2.csv", skiprows=1, header=None
# )
# xy_data_total = pd.read_csv("PR_complete_DOE_results_restructure_v2.csv")

# xy_data_validation = pd.read_csv("PR_validation_complete_data_resturcture.csv")
xy_data_validation = pd.read_csv("PR_validation_single_restrucutre.csv")

input_data = xy_data.iloc[:, :4]
output_data = xy_data.iloc[:, 4:]

# Define labels, and split training and validation data
# note that PySMO requires that labels are passed as string lists
input_labels = list(input_data.columns)
# input_labels = [str(i) for i in list(input_data.columns)]
output_labels = list(output_data.columns)
# output_labels = [str(i) for i in list(output_data.columns)]

# n_data = xy_data[input_labels[0]].size
# data_training, data_validation = split_training_validation(
#     xy_data, 0.8, seed=n_data
# )  # seed=100

data_training = xy_data
data_validation = xy_data_validation

# Create PySMO trainer object
surrogate_trainer = PysmoRBFTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

# Set PySMO options
surrogate_trainer.config.basis_function = "gaussian"
surrogate_trainer.config.regularization = True

# Train surrogate (calls PySMO through IDAES Python wrapper)
rbf_train = surrogate_trainer.train_surrogate()

# create callable surrogate object
xmin, xmax = [-1.3, 0.05, 5, 5], [-0.8, 0.15, 45, 30]
# input_bounds = {input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))}
# rbf_surr = PysmoSurrogate(rbf_train, input_labels, output_labels, input_bounds)
rbf_surr = PysmoSurrogate(rbf_train, input_labels, output_labels)

# save model to JSON
model = rbf_surr.save_to_file("pysmo_RBF_PR_surrogate.json", overwrite=True)

metrics_training = compute_fit_metrics(rbf_surr, data_training)
metrics_validation = compute_fit_metrics(rbf_surr, data_validation)

# visualize with IDAES surrogate plotting tools
# surrogate_scatter2D(rbf_surr, data_training, filename='pysmo_poly_train_scatter2D.pdf')
surrogate_parity(rbf_surr, data_training, filename="pysmo_poly_train_parity.pdf")
# surrogate_residual(rbf_surr, data_training, filename='pysmo_poly_train_residual.pdf')

# visualize with IDAES surrogate plotting tools
# surrogate_scatter2D(rbf_surr, data_validation, filename='pysmo_poly_val_scatter2D.pdf')
surrogate_parity(rbf_surr, data_validation, filename="pysmo_poly_val_parity.pdf")
# surrogate_residual(rbf_surr, data_validation, filename='pysmo_poly_val_residual.pdf')
