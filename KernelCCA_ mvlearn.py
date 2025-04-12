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

# Load the data
data = pd.read_csv("Connectivity_Indexed.csv").dropna()
data2 = pd.read_csv('Filtered_Participant_SNP.csv')

# Define the feature columns
# X = data[['DMN 1 ', 'DMN 2 ', 'SMN 1', 'SMN 2', 'SMN 3 ', 'SMN 4', 'SMN 5', 'SMN 6', 'SMN 7', 'SMN 8', 'SMN 9',       
                   #'DAN 2 ', 'DAN 3', 'DAN 4', 'DAN 5', 'DAN 6', 'DAN 7', 'DAN 8', 'DAN 10', 'DAN 11', 'CN 1 ', 'CN 2 ', 'CN 3',
                   #'CN 4', 'CN 5', 'CN 6', 'VN 1 ', 'VN 2 ', 'VN 3', 'VN 4', 'VN 6', 'VN 7', 'VN 9', 'VN 10']]

Y = data[['DAN 7', 'SMN 12', 'CN 4', 'VN 7', 'VN 10', 'SMN 9', 'SMN 2', 'DAN 3', 'CN 1 ', 'DAN 2 ']]

# Remove rows with NaN values
data = data.dropna()

n_views = 2

# Ensure indices are aligned before dropping NaNs
X = data2[['rs28394864_A', 'rs602602_A', 'rs12151021_A', 'rs429358_C', 'rs1354106_G', 'rs4663105_C', 
           'rs6069737_T', 'rs679515_T', 'rs1582763_A', 'rs1532278_A', 'rs561655_G', 'rs11218343_G', 
           'rs7146179_A', 'rs12590654_A', 'rs1846190_A', 'rs187370608_A', 'rs9369716_T', 'rs7912495_G', 
           'rs7384878_C', 'rs3935067_C']]

data3 = pd.read_excel('20_SNP.xlsx') 

# Rename SNPs in X based on data3
for i, col in enumerate(Y.columns):
    match = data3[data3['Lead Variant'].str[:6] == col[:6]]
    if not match.empty:
        Y.columns.values[i] = match['Gene'].values[0]


# Drop NaN rows to align the datasets
X = X.dropna()
print(f"Shape of data after filtering: {data.shape}")
print(f"Shape of data2 after filtering: {data2.shape}")

# Standardize both datasets
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)


# Perform Kernel CCA
n_components = 10
kmcca = KMCCA(n_components=n_components, kernel="linear")
try:
    X_c, Y_c = kmcca.fit_transform([X_scaled, Y_scaled])
except ValueError as e:
    print(f"Error during Kernel CCA computation: {e}")
    sys.exit(1)


# Compute correlation scores
correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
print("Canonical Correlations:")
print(correlations)


# Compute p-values for the canonical correlations
# Compute Wilks' Lambda and p-values for each mode
n_samples = X_scaled.shape[0]
p_values = []

for i in range(n_components):
    # Compute Wilks' Lambda for the current mode
    wilks_lambda = np.prod([1 - corr**2 for corr in correlations[i:]])
    
    # Compute the test statistic and p-value
    df = 2 * (n_components - i)  # Degrees of freedom for the current mode
    test_statistic = -n_samples * np.log(wilks_lambda)
    p_value = 1 - chi2.cdf(test_statistic, df)
    p_values.append(p_value)

# Print all p-values for the top 10 modes
print("P-values for the top 10 modes:")
for i, p_val in enumerate(p_values):
    print(f"Mode {i+1}: {p_val}")


# Plot the first 3 canonical variables
plt.figure(figsize=(18, 6))
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'wspace': 0.3})  # Equal spacing with wspace

for i in range(3):
    axs[i].scatter(X_c[:, i], Y_c[:, i], alpha=0.5)
    axs[i].set_title(f"Mode {i+1}")
    axs[i].set_xlabel("X_c")
    axs[i].set_ylabel("Y_c")

plt.show()




