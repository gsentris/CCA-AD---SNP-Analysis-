import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("Connectivity_Indexed.csv").dropna()
data2 = pd.read_csv('Filtered_Participant_SNP.csv')

# Remove rows with NaN values
data = data.dropna()

# Ensure indices are aligned before dropping NaNs
Y = data[['DAN 7', 'SMN 12', 'CN 4', 'VN 7', 'VN 10', 'SMN 9', 'SMN 2', 'DAN 3', 'CN 1 ', 'DAN 2 ']]
X = data2[['rs28394864_A', 'rs602602_A', 'rs12151021_A', 'rs429358_C', 'rs1354106_G', 'rs4663105_C', 
           'rs6069737_T', 'rs679515_T', 'rs1582763_A', 'rs1532278_A', 'rs561655_G', 'rs11218343_G', 
           'rs7146179_A', 'rs12590654_A', 'rs1846190_A', 'rs187370608_A', 'rs9369716_T', 'rs7912495_G', 
           'rs7384878_C', 'rs3935067_C']]

data3 = pd.read_excel('20_SNP.xlsx')

# Rename SNPs in X based on data3
for i, col in enumerate(X.columns):
    match = data3[data3['Lead Variant'].str[:6] == col[:6]]
    if not match.empty:
        X.columns.values[i] = match['Gene'].values[0]


# Drop NaN rows to align the datasets
Y = Y.dropna()
print(f"Shape of data after filtering: {data.shape}")
print(f"Shape of data2 after filtering: {data2.shape}")

# Standardize both datasets
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Perform Canonical Correlation Analysis (CCA)
cca = CCA(n_components=min(X.shape[1], Y.shape[1]))
X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

# Compute canonical correlations
correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(cca.n_components)]
r_squared = [corr ** 2 for corr in correlations]

print("Canonical Correlations:", correlations)
print("R-squared values:", r_squared)

pve_x = np.var(X_c, axis=0) / np.var(X_scaled, axis=0).sum()
pve_y = np.var(Y_c, axis=0) / np.var(Y_scaled, axis=0).sum()

print(f"Proportion of Variance Explained in X: {pve_x.sum():.4f}")
print(f"Proportion of Variance Explained in Y: {pve_y.sum():.4f}")


# Function to perform Wilk’s Lambda significance test
def canonical_correlation_test(canonical_corrs, N, px, py):
    """
    Perform Wilk's Lambda significance test for canonical correlations.
    
    Parameters:
    - canonical_corrs: List of canonical correlations (gamma values).
    - N: Sample size.
    - px: Number of X variables.
    - py: Number of Y variables.
    
    Returns:
    - List of tuples (Wilk's lambda, Chi-square statistic, degrees of freedom, p-value)
    """
    m = min(px, py)
    lambdas = [(1 / (1 - gamma**2)) - 1 for gamma in canonical_corrs]

    wilks = np.zeros(m)
    wilks[-1] = 1 / (1 + lambdas[-1])
    for j in range(m-2, -1, -1):
        wilks[j] = (1 / (1 + lambdas[j])) * wilks[j + 1]

    results = []
    for j in range(m):
        df_j = (px - j) * (py - j)
        chi_sq_j = -np.log(wilks[j]) * (N - (px + py + 3) / 2)
        p_value = 1 - chi2.cdf(chi_sq_j, df_j)

        results.append((wilks[j], chi_sq_j, df_j, p_value))
        print(f"Canonical Correlation {j+1}: Wilk's Lambda={wilks[j]:.4f}, "
              f"Chi-Square={chi_sq_j:.4f}, df={df_j}, p-value={p_value:.4f}")
    
    return results

# Perform significance test
N = X.shape[0]
px, py = X.shape[1], Y.shape[1]
canonical_correlation_test(correlations, N, px, py)
# print the canonical correlations with the p-values in csv file 
canonical_correlations = pd.DataFrame({'Canonical Correlation': correlations, 'p-value': [result[3] for result in canonical_correlation_test(correlations, N, px, py)]})
canonical_correlations.to_csv('canonical_correlations.csv', index=False)

# Save the canonical variables to a CSV file
canonical_vars = pd.DataFrame(np.hstack((X_c, Y_c)), 
                              columns=[f'X_c{i+1}' for i in range(X_c.shape[1])] + 
                                      [f'Y_c{i+1}' for i in range(Y_c.shape[1])])
canonical_vars.to_csv('canonical_variables.csv', index=False)


# Plot the first canonical variables
plt.scatter(X_c[:, 0], Y_c[:, 0])
plt.xlabel('First Canonical Variable of X')
plt.ylabel('First Canonical Variable of Y')
plt.title(f'Scatterplot of First Canonical Variables\n(R-squared = {r_squared[0]:.3f})')
plt.grid()
plt.show()

# Print CCA coefficients
print("CCA Coefficients for X:")
print(cca.x_weights_)
print("\nCCA Coefficients for Y:")
print(cca.y_weights_)

# Visualize canonical weights
plt.figure(figsize=(12, 6))

# Plot CCA coefficients for X (SNPs)
plt.subplot(1, 2, 1)
plt.barh(np.arange(X.shape[1]), cca.x_weights_[:, 0], align='center')
plt.yticks(np.arange(X.shape[1]), X.columns)
plt.xlabel('Canonical Weights (X)')
plt.title('Canonical Weights for SNPs')

# Plot CCA coefficients for Y (Networks)
plt.subplot(1, 2, 2)
plt.barh(np.arange(Y.shape[1]), cca.y_weights_[:, 0], align='center')
plt.yticks(np.arange(Y.shape[1]), Y.columns)
plt.xlabel('Canonical Weights (Y)')
plt.title('Canonical Weights for Networks')
plt.tight_layout()
plt.show()

# Heatmap for SNP Weights
sns.heatmap(cca.x_weights_, annot=True, cmap='coolwarm', xticklabels=[f'Canonical {i+1}' for i in range(cca.n_components)], 
            yticklabels=X.columns)
plt.title('SNP Weights in Canonical Variables')
plt.xlabel('Canonical Variables')
plt.ylabel('SNPs')
plt.show()

# Heatmap for Network Weights
sns.heatmap(cca.y_weights_, annot=True, cmap='coolwarm', xticklabels=[f'Canonical {i+1}' for i in range(cca.n_components)], 
            yticklabels=Y.columns)
plt.title('Network Weights in Canonical Variables')
plt.xlabel('Canonical Variables')
plt.ylabel('Networks')
plt.show()

#plot rest of the modes to see if there's any significance
for i in range(1, cca.n_components):
    plt.scatter(X_c[:, i], Y_c[:, i])
    plt.xlabel(f'Canonical Variable {i+1} of X')
    plt.ylabel(f'Canonical Variable {i+1} of Y')
    plt.title(f'Scatterplot of Canonical Variables {i+1}\n(R-squared = {r_squared[i]:.3f})')
    plt.grid()
    plt.show()
