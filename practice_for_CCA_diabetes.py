import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_linnerud,load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from mvlearn.plotting.plot import crossviews_plot
from scipy.stats import chi2

# Load the dataset
data = load_diabetes()
#X data I want to be columns 1-4
#Y data I want to be columns 5-11 
X = data['data'][:, :4]  # Shape: (442, 4)
y = data['data'][:, 4:]  # Shape: (442, 7)
#X = data['data'][:, :3]  # Shape: (442, 3)

print(y.shape)

# append target to the y data to the end of the y data
y = np.append(y, data['target'].reshape(-1, 1), axis=1)  # Shape: (442, 8)
# append column 6 of the y data to the end of the x data
X = np.append(X, y[:, 6].reshape(-1, 1), axis=1)  # Shape: (442, 5)
#remove y6 from y data
y = np.delete(y, 6, axis=1)  # Shape: (442, 7)
# print the shape of x and y data
print(X.shape)

print(y.shape)

 
# name columns for x and y as the following 
#x1= age, x2= sex, x3= bmi, x4=bp, x5 = blood sugar
#y1= serum cholesterol , y2= ld lipoproteins, y3= hd density lipoproteins, y4= tital cholesterol , y5= triglycerides , y6= measure of diabetes progression,
X_columns = ['Age', 'Sex', 'BMI', 'BP', 'Blood Sugar']
Y_columns = ['Serum Cholesterol', 'LD Lipoproteins', 'HD Lipoproteins', 
             'Total Cholesterol', 'Triglycerides', 'Diabetes Progression']

X_df = pd.DataFrame(X, columns=X_columns)
Y_df = pd.DataFrame(y, columns=Y_columns)

# Standardize both views
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(y)

# Perform CCA
n_components = 5
cca = CCA(n_components=n_components)
view1_c, view2_c = cca.fit_transform(X_scaled, Y_scaled)

# Canonical Correlations & R^2
correlations = [np.corrcoef(view1_c[:, i], view2_c[:, i])[0, 1] for i in range(cca.n_components)]
r_squared = [corr ** 2 for corr in correlations]

print("Canonical Correlations:", correlations)
print("R² values:", r_squared)

# Compute R² score
r2 = r2_score(view2_c, view1_c)
print("Overall R² Score:", r2)

# Convert to DataFrames
X_df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
Y_df = pd.DataFrame(Y_scaled, columns=[f'Target {i+1}' for i in range(Y_scaled.shape[1])])

# Scatter plot of the canonical variables with equal spacing
plt.figure(figsize=(18, 6))  # Adjust the figure size for better spacing
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'wspace': 0.3})  # Equal spacing with wspace

# First Canonical Variable
axs[0].scatter(view1_c[:, 0], view2_c[:, 0])
axs[0].set_xlabel('First Canonical Variable of X')
axs[0].set_ylabel('First Canonical Variable of Y')
axs[0].set_title(f'Scatterplot of First Canonical Variables\n(R-squared = {r_squared[0]:.3f})')

# Second Canonical Variable
axs[1].scatter(view1_c[:, 1], view2_c[:, 1])
axs[1].set_xlabel('Second Canonical Variable of X')
axs[1].set_ylabel('Second Canonical Variable of Y')
axs[1].set_title(f'Scatterplot of Second Canonical Variables\n(R-squared = {r_squared[1]:.3f})')

# Third Canonical Variable
axs[2].scatter(view1_c[:, 2], view2_c[:, 2])
axs[2].set_xlabel('Third Canonical Variable of X')
axs[2].set_ylabel('Third Canonical Variable of Y')
axs[2].set_title(f'Scatterplot of Third Canonical Variables\n(R-squared = {r_squared[2]:.3f})')

plt.tight_layout()  # Automatically adjust subplot parameters for better spacing
plt.show()
# Bar plot of CCA coefficients
plt.figure(figsize=(12, 6))

# Plot for X loadings
plt.subplot(1, 2, 1)
plt.barh(np.arange(X.shape[1]), cca.x_weights_[:, 0], align='center')
plt.yticks(np.arange(X.shape[1]), X_columns)  # Use actual column names for X
plt.xlabel('Canonical Loadings for X')
plt.title('Canonical Loadings for X')

# Plot for Y loadings
plt.subplot(1, 2, 2)
plt.barh(np.arange(Y_scaled.shape[1]), cca.y_weights_[:, 0], align='center')
plt.yticks(np.arange(Y_scaled.shape[1]), Y_columns)  # Use actual column names for Y
plt.xlabel('Canonical Loadings for Y')
plt.title('Canonical Loadings for Y')

plt.tight_layout()
plt.show()

# Heatmap for X weights
sns.heatmap(cca.x_weights_, annot=True, cmap='coolwarm', 
            xticklabels=[f'Canonical {i+1}' for i in range(cca.n_components)], 
            yticklabels=X_columns)  # Use actual column names for X
plt.title('Canonical Weights in X')
plt.xlabel('Canonical Variables')
plt.ylabel('Features')
plt.show()

# Heatmap for Y weights
sns.heatmap(cca.y_weights_, annot=True, cmap='coolwarm', 
            xticklabels=[f'Canonical {i+1}' for i in range(cca.n_components)], 
            yticklabels=Y_columns)  # Use actual column names for Y
plt.title('Canonical Weights in Y')
plt.xlabel('Canonical Variables')
plt.ylabel('Targets')
plt.show()

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

N = X.shape[0]
px, py = X.shape[1], y.shape[1]
canonical_correlation_test(correlations, N, px, py)
