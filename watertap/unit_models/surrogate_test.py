from idaes.core.surrogate.pysmo import radial_basis_function, polynomial_regression
import pandas as pd

# Load dataset from a csv file
xy_data = pd.read_csv("PR_DOE_unique_data_restructure_v2.csv", header=1, index_col=0)

# Initialize the RadialBasisFunctions class, extract the list of features and train the model
rbf_init = radial_basis_function.RadialBasisFunctions(xy_data)
features = rbf_init.get_feature_vector()
rbf_fit = rbf_init.training()

# Create a python list from the headers of the dataset supplied for training
list_vars = []
for i in features.keys():
    list_vars.append(features[i])

# Pass list to generate_expression function to obtain a Pyomo expression as output
print(rbf_fit.generate_expression(list_vars))

# Create a python list from the headers of the dataset supplied for training
y_unsampled = rbf_init.predict_output(rbf_fit, xy_data)
