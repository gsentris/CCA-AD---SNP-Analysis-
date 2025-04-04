import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Load the dataset
data = pd.read_csv("mmreg.csv")
# X data I want to be columns 1-4
# Y data I want to be columns 5-11 
X = data[['locus_of_control','self_concept','motivation']]
Y = data[['read', 'write', 'math', 'science','female']]  # Shape: (442, 4)


# Load the data
# Remove rows with NaN values
data = data.dropna()

# Perform Z-normalization (standardization) on both datasets
X_scaled = (X - X.mean()) / X.std()
Y_scaled = (Y - Y.mean()) / Y.std()

# Perform Canonical Correlation Analysis (CCA)
cca = CCA(n_components=3)
X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

# Compute canonical x_weights and y_weights 
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


# Plot all canonical variables into one figure
plt.figure(figsize=(12, 8))
for i in range(cca.n_components):
    plt.subplot(2, (cca.n_components + 1) // 2, i + 1)
    plt.scatter(X_c[:, i], Y_c[:, i])
    plt.xlabel(f'Canonical Variable {i+1} of X')
    plt.ylabel(f'Canonical Variable {i+1} of Y')
    plt.title(f'Canonical Variables {i+1}\n(R-squared = {r_squared[i]:.3f})')
    plt.grid()

plt.tight_layout()
plt.show()

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
sns.heatmap(cca.x_loadings_, annot=True, cmap='coolwarm', xticklabels=[f'Canonical {i+1}' for i in range(cca.n_components)], 
            yticklabels=X.columns)
plt.title('SNP Weights in Canonical Variables')
plt.xlabel('Canonical Variables')
plt.ylabel('SNPs')
plt.show()

# Heatmap for Network Weights
sns.heatmap(cca.y_loadings_, annot=True, cmap='coolwarm', xticklabels=[f'Canonical {i+1}' for i in range(cca.n_components)], 
            yticklabels=Y.columns)
plt.title('Network Weights in Canonical Variables')
plt.xlabel('Canonical Variables')
plt.ylabel('Networks')
plt.show()

#Get canonical correlation loadings 
# Loadings for X


# Plot the approximate canonical loadings
plt.figure(figsize=(12, 6))


x_loadings = cca.x_loadings_

print("X Loadings:")
print(x_loadings)

# calculate and plot the correlations of all components
corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(cca.n_components)]    
plt.plot(corrs)
plt.xlabel('cca_idx')
plt.ylabel('cca_corr')
plt.show()

r, _ = pearsonr(np.squeeze(X.iloc[:, 0]), np.squeeze(X_c[:, 0]))
print(f"Pearson correlation between X and first canonical variable: {r:.4f}")
print(r)




