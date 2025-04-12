import numpy as np
import sys
sys.path.append("../../..")

from mvlearn.embed import KMCCA
from mvlearn.plotting.plot import crossviews_plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy import stats
import warnings
from matplotlib import MatplotlibDeprecationWarning
from scipy.stats import chi2
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
from sklearn.linear_model import MultiTaskLassoCV
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("Connectivity_Indexed.csv").dropna()
data2 = pd.read_csv('Filtered_Participant_SNP.csv')

# Define the feature columns
X = data[['DMN 1 ', 'DMN 2 ', 'SMN 1', 'SMN 2', 'SMN 3 ', 'SMN 4', 'SMN 5', 'SMN 6', 'SMN 7', 'SMN 8', 'SMN 9',       
          'DAN 2 ', 'DAN 3', 'DAN 4', 'DAN 5', 'DAN 6', 'DAN 7', 'DAN 8', 'DAN 10', 'DAN 11', 'CN 1 ', 'CN 2 ', 'CN 3',
          'CN 4', 'CN 5', 'CN 6', 'VN 1 ', 'VN 2 ', 'VN 3', 'VN 4', 'VN 6', 'VN 7', 'VN 9', 'VN 10']]


# Remove rows with NaN values
data = data.dropna()

n_views = 2

# Ensure indices are aligned before dropping NaNs
Y = data2[['rs28394864_A', 'rs602602_A', 'rs12151021_A', 'rs429358_C', 'rs1354106_G', 'rs4663105_C', 
           'rs6069737_T', 'rs679515_T', 'rs1582763_A', 'rs1532278_A', 'rs561655_G', 'rs11218343_G', 
           'rs7146179_A', 'rs12590654_A', 'rs1846190_A', 'rs187370608_A', 'rs9369716_T', 'rs7912495_G', 
           'rs7384878_C', 'rs3935067_C']]




data3 = pd.read_excel('20_SNP.xlsx') 



# Rename SNPs in X based on data3
for i, col in enumerate(Y.columns):
    match = data3[data3['Lead Variant'].str[:6] == col[:6]]
    if not match.empty:
        Y.columns.values[i] = match['Gene'].values[0]
    #however remove all the SNPs in data3[Rank] over 10

for i, col in enumerate(Y.columns):
    if col in data3[data3['Rank'] > 20]['Gene'].values:
        Y = Y.drop(columns=col)




# Drop NaN rows to align the datasets
X = X.dropna()
print(f"Shape of data after filtering: {X.shape}")
print(f"Shape of data2 after filtering: {Y.shape}")

# Standardize both datasets
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

from sklearn.linear_model import LassoCV
import numpy as np

# I want to write True or false for lasso for choosing when to use it
# Lasso regression for feature selection
#if lasso(MultiTaskLassoCV) true run otherwise don't run 
use_lasso  = True # Set to True to use Lasso, False to skip

if use_lasso:
    # Run Lasso with cross-validation
    lasso = MultiTaskLassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, Y_scaled)

    # Get absolute coefficient magnitudes
    coef_magnitudes = np.abs(lasso.coef_).sum(axis=0)  # Sum across targets

    # Select the top 10-15 features
    top_k = 10  # Change this number as needed
    top_features_idx = np.argsort(coef_magnitudes)[-top_k:]  # Indices of top features
    selected_features = X.columns[top_features_idx]

    print("Selected Features:", list(selected_features))

    # Keep only selected features
    X_selected = X_scaled[:, top_features_idx]
else:
    # If Lasso is not used, keep all features
    X_selected = X_scaled
    selected_features = X.columns

# Perform Kernel CCA
n_components = 10
kmcca = KMCCA(n_components=n_components, kernel="linear")
try:
    X_c, Y_c = kmcca.fit_transform([X_selected, Y_scaled])
except ValueError as e:
    print(f"Error during Kernel CCA computation: {e}")
    sys.exit(1)


# Compute correlation scores
correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
print("Canonical Correlations:")
print(correlations)

# Compute p-values for the canonical correlations
# Compute Wilks' Lambda and p-values for each mode
n_samples = X_selected.shape[0]
p_values = []

def compute_p_values(correlations, n_components, n_samples):
    """
    Compute p-values for canonical correlations using Wilks' Lambda.

    Parameters:
    correlations (list): List of canonical correlations.
    n_components (int): Number of components in the analysis.
    n_samples (int): Number of samples in the dataset.

    Returns:
    list: P-values for each mode.
    """
    p_values = []
    for i in range(n_components):
        # Compute Wilks' Lambda for the current mode
        wilks_lambda = np.prod([1 - corr**2 for corr in correlations[i:]])
        
        # Compute the test statistic and p-value
        df = 2 * (n_components - i)  # Degrees of freedom for the current mode
        test_statistic = -n_samples * np.log(wilks_lambda)
        p_value = 1 - chi2.cdf(test_statistic, df)
        p_values.append(p_value)
    return p_values

# Compute p-values for the canonical correlations
p_val = compute_p_values(correlations, n_components, n_samples)
print("P-values for the canonical correlations:")
print(p_val)




# Permutation test on the dataset 
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances

# === Permutation Test === #
n_permutations = 100 
permutation_correlations = []

print("\nRunning permutation test...")
for i in range(n_permutations):
    # Shuffle Y across samples
    Y_shuffled = shuffle(Y_scaled, random_state=i)
    
    try:
        kmcca = KMCCA(n_components=n_components, kernel="linear")
        X_c_perm, Y_c_perm = kmcca.fit_transform([X_selected, Y_shuffled])
        perm_corrs = [np.corrcoef(X_c_perm[:, j], Y_c_perm[:, j])[0, 1] for j in range(n_components)]
        permutation_correlations.append(perm_corrs)
    except Exception as e:
        print(f"Permutation {i} failed: {e}")
        continue

# Convert to numpy array for easier computation
permutation_correlations = np.array(permutation_correlations)

# Compute empirical p-values
empirical_p_values = []
for j in range(n_components):
    obs_corr = correlations[j]
    greater_count = np.sum(permutation_correlations[:, j] >= obs_corr)
    emp_pval = (greater_count + 1) / (n_permutations + 1)  # add 1 for stability
    empirical_p_values.append(emp_pval)


print("\nEmpirical P-values from Permutation Test:")
for i, p_val in enumerate(empirical_p_values):
    print(f"Mode {i+1}: {p_val:.4f}")

print("\nPermutation Correlations:")
print(perm_corrs)



