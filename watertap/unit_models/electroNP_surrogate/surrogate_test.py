from idaes.core.surrogate.pysmo import radial_basis_function, polynomial_regression
import pandas as pd
from idaes.core.surrogate.pysmo import sampling as sp

# Load dataset from a csv file
xy_data = pd.read_csv("PR_DOE_unique_data_restructure_v2.csv", header=None)
# xy_data = pd.read_csv("PR_complete_DOE_results_restructure_v2.csv", header=1, index_col=None)

# Initialize the CVT sampling method and generate 25 samples
space_init = sp.CVTSampling(xy_data, sampling_type="selection", number_of_samples=25)
samples = space_init.sample_points()

# Initialize the RadialBasisFunctions class, extract the list of features and train the model
rbf_init = radial_basis_function.RadialBasisFunctions(
    samples, basis_function="gaussian"
)
features = rbf_init.get_feature_vector()
rbf_fit = rbf_init.training()
rbf_fit.print_report()
rbf_fit.parity_residual_plots()


# Create a python list from the headers of the dataset supplied for training
list_vars = []
for i in features.keys():
    list_vars.append(features[i])

# Pass list to generate_expression function to obtain a Pyomo expression as output
print(rbf_fit.generate_expression(list_vars))

# Create a python list from the headers of the dataset supplied for training
y_unsampled = rbf_init.predict_output(xy_data.values[:, 0:-1])
