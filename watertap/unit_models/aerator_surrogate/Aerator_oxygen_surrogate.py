import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # suppress interactive window in PyCharm IPython console

import warnings

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

from idaes.core.surrogate.pysmo_surrogate import (
    PysmoPolyTrainer,
    PysmoSurrogate,
)
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.plotting.sm_plotter import surrogate_parity
from idaes.core.surrogate.metrics import compute_fit_metrics

# 1. Load and prepare data
raw = pd.read_csv("oxygen_data.csv")

# Keep only the two inputs and one output
df = raw[["Immersion Depth (in)", "Capacity", "Oxygen Mass Flowrate(lb/hr)"]].copy()

input_labels = ["Immersion Depth (in)", "Capacity"]
output_labels = ["Oxygen Mass Flowrate(lb/hr)"]

# 2. Train / validation split (80/20)
n_data = len(df)
data_training, data_validation = split_training_validation(df, 0.8, seed=n_data)

# 3. Set up polynomial trainer
surrogate_trainer = PysmoPolyTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

surrogate_trainer.config.maximum_polynomial_order = (
    2  # linear in each var + interaction
)
surrogate_trainer.config.multinomials = True  # include cross term d*C
surrogate_trainer.config.training_split = 0.8

# 4. Train
poly_train = surrogate_trainer.train_surrogate()

# 5. Build callable surrogate with input bounds
xmin = [-5.12, 50.0]  # [min immersion depth (in), min capacity (%)]
xmax = [6.02, 100.0]  # [max immersion depth (in), max capacity (%)]
input_bounds = {input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))}

poly_surr = PysmoSurrogate(poly_train, input_labels, output_labels, input_bounds)

# 6. Save surrogate to JSON
poly_surr.save_to_file("aerator_oxygen_surrogate.json", overwrite=True)

# 7. Compute and print fit metrics
metrics_training = compute_fit_metrics(poly_surr, data_training)
metrics_validation = compute_fit_metrics(poly_surr, data_validation)


def print_metrics(label, metrics):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for output, m in metrics.items():
        print(f"  Output : {output}")
        print(f"  {'─'*45}")
        print(f"  {'R²':<20} {m['R2']:.6f}")
        print(f"  {'RMSE (lb/hr)':<20} {m['RMSE']:.6f}")
        print(f"  {'MAE  (lb/hr)':<20} {m['MAE']:.6f}")
        print(f"  {'Max AE (lb/hr)':<20} {m['maxAE']:.6f}")
        print(f"  {'MSE  (lb/hr)²':<20} {m['MSE']:.6f}")
        print(f"  {'SSE  (lb/hr)²':<20} {m['SSE']:.6f}")
    print(f"{'='*55}\n")


print_metrics("Training Metrics", metrics_training)
print_metrics("Validation Metrics", metrics_validation)

# 8. Parity plots (saved to PDF, no interactive window)
surrogate_parity(poly_surr, data_training, filename="aerator_oxygen_train_parity.pdf")
plt.close("all")
surrogate_parity(poly_surr, data_validation, filename="aerator_oxygen_val_parity.pdf")
plt.close("all")
