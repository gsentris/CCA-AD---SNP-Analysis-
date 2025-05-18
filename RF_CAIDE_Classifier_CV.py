import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import normaltest

# Load the data
data = pd.read_csv("Connectivity_Indexed.csv")

# Remove rows with NaN values
data = data.dropna()

# Print column names to verify
print("Column names in the DataFrame:")
print(data.columns)

# Convert 'caide_score' to numeric values
data['caide_score'] = pd.to_numeric(data['caide_score'], errors='coerce')

# Reclassify 'caide_score' into low-risk (0-6) and high-risk (7-15)
data['caide_risk'] = data['caide_score'].apply(lambda x: 0 if x <= 6 else 1)

# Count the number of participants in each group
risk_counts = data['caide_risk'].value_counts()
print(f"Number of participants in low risk: {risk_counts[0]}")
print(f"Number of participants in high risk: {risk_counts[1]}")

# Define the feature columns

feature_columns = data[["DMN 1",'SN 1','SMN 8','DAN 5','CN 4','VN 6']]

# Ensure the feature columns exist in the DataFrame
feature_columns = [col for col in feature_columns if col in data.columns]

# Define feature matrix X and target vector y
X = data[feature_columns]
y = data['caide_risk']

# Split the data (50% training, 25% validation, 25% testing)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_centered = X_train - X_train.mean()
X_train_scaled = X_centered / X_train.std()
X_val_centered = X_val - X_train.mean()
X_val_scaled = X_val_centered / X_train.std()
X_test_centered = X_test - X_train.mean()
X_test_scaled = X_test_centered / X_train.std()



# Initialize Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Feature selection using RFE (selecting top 20 features)
selector = RFE(estimator=rf_model, step=1)
selector.fit(X_train_scaled, y_train)

# Get selected feature names
selected_features = X.columns[selector.support_]

# Transform datasets using selected features
X_train_selected = selector.transform(X_train_scaled)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Print selected features
print("Selected Features using RFE:")
print(list(selected_features))

# Define parameter grid for Random Forest
param_grid = {'max_depth': list(range(3, 20, 2))}

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

# Best max_depth value
best_depth = grid_search.best_params_['max_depth']
print(f"Optimal max_depth: {best_depth}")

# Train best model
best_rf = RandomForestClassifier(max_depth=best_depth, random_state=42, n_estimators=100)
best_rf.fit(X_train_selected, y_train)

# Evaluate on validation set
y_val_pred = best_rf.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
val_classification_report_str = classification_report(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Confusion Matrix:\n{val_conf_matrix}")
print(f"Classification Report:\n{val_classification_report_str}")

# Feature Importance Plot
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importance After RFE")
plt.bar(range(len(selected_features)), importances[:len(selected_features)], align='center')
plt.xticks(range(len(selected_features)), selected_features, rotation=90)
plt.tight_layout()
plt.show()

# ROC AUC Curve
y_val_prob = best_rf.predict_proba(X_val_selected)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
roc_auc = roc_auc_score(y_val, y_val_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Find the best threshold
gmeans = np.sqrt(tpr * (1 - fpr))
ix = np.argmax(gmeans)
best_threshold = thresholds[ix]
print(f'Best Threshold: {best_threshold:.2f}')

# Apply the best threshold
y_val_pred_best_threshold = (y_val_prob >= best_threshold).astype(int)
val_accuracy_best_threshold = accuracy_score(y_val, y_val_pred_best_threshold)
val_conf_matrix_best_threshold = confusion_matrix(y_val, y_val_pred_best_threshold)
val_classification_report_best_threshold_str = classification_report(y_val, y_val_pred_best_threshold)

# Calculate specificity
tn, fp, fn, tp = val_conf_matrix_best_threshold.ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.2f}")

# Precision-Recall AUC Curve
precision, recall, pr_thresholds = precision_recall_curve(y_val, y_val_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Save results to Excel
report_dict = classification_report(y_val, y_val_pred_best_threshold, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel("validation_classification_report.xlsx", index=True)

results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'Specificity', 'F1-Score'],
    'Score': [
        precision_score(y_val, y_val_pred_best_threshold),
        recall_score(y_val, y_val_pred_best_threshold),
        specificity,
        f1_score(y_val, y_val_pred_best_threshold)
    ]
})
results_df.to_excel("validation_results_with_specificity.xlsx", index=False)

print("Results saved successfully!")
