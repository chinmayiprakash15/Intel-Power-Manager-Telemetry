# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Intel Power Gadget data
csv_file_path = '/content/DataSet.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the data to understand its structure
print(data.head())

# Preprocessing based on the correct column names
# Handling missing values
data.fillna(method='ffill', inplace=True)

# Feature Engineering
# Example feature: Rolling average of the CPU Utilization over the last 5 readings
data['cpu_util_rolling_avg'] = data[' CPU Utilization(%)'].rolling(window=5).mean().fillna(data[' CPU Utilization(%)'].mean())

# Ensure no NaN values remain
data.fillna(method='bfill', inplace=True)  # Use backward fill as a safeguard

# Select features and target
features = [' CPU Utilization(%)', 'CPU Frequency_0(MHz)', 'Processor Power_0(Watt)', 'GT Power_0(Watt)', 'cpu_util_rolling_avg']
target = 'Failure'

# Create a synthetic 'Failure' column for demonstration purposes
data[target] = np.random.binomial(1, 0.1, len(data))

X = data[features]
y = data[target]

# Verify no NaN values are present in the features
print(X.isna().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Development
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Model Evaluation
# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC Score: {roc_auc:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importance = rf_model.feature_importances_
features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importance')
plt.show()

# Deployment (Pseudo-code)
# Real-time monitoring would involve continuously updating the model with new data
# For demonstration, we simulate real-time data updates with a loop (Replace this with actual real-time data handling in practice)
import time

for i in range(5):  # Simulating 5 new data points
    new_data = pd.DataFrame({
        ' CPU Utilization(%)': [12 + np.random.normal()],
        'CPU Frequency_0(MHz)': [1300 + np.random.normal()],
        'Processor Power_0(Watt)': [5 + np.random.normal()],
        'GT Power_0(Watt)': [0.3 + np.random.normal()],
        'cpu_util_rolling_avg': [12 + np.random.normal()]
    })

    prediction = rf_model.predict(new_data)
    print(f'New Data Point {i+1}:', new_data)
    print(f'Predicted Failure: {prediction[0]}')
    time.sleep(2)  # Simulate time delay
