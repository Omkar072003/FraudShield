import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("fraud_data.csv")

# Identify target column (assuming 'fraud' in name)
target_col = next((col for col in data.columns if 'fraud' in col.lower()), None)
if not target_col:
    raise ValueError("No target column found (e.g., 'fraud', 'is_fraud')")

# Feature engineering
if 'time' in data.columns:
    data['hour_of_day'] = pd.to_datetime(data['time'], errors='coerce').dt.hour.fillna(np.random.randint(0, 24))
    data['day_of_week'] = pd.to_datetime(data['time'], errors='coerce').dt.dayofweek.fillna(np.random.randint(0, 7))
else:
    data['hour_of_day'] = np.random.randint(0, 24, size=len(data))
    data['day_of_week'] = np.random.randint(0, 7, size=len(data))

data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Log transform numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != target_col and data[col].min() >= 0:
        data[f'{col}_log'] = np.log1p(data[col])

# Risk scoring for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    risk_map = {val: idx / len(data[col].unique()) + 0.3 for idx, val in enumerate(data[col].unique())}
    data[f'{col}_risk'] = data[col].map(risk_map)

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

# Define features and target
feature_columns = [col for col in data_encoded.columns if col != target_col]
X = data_encoded[feature_columns]
y = data_encoded[target_col]

# Save feature list for API consistency
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Grid search parameters
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

# Train model with grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

# Best model
best_pipeline = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate model
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Feature importance
feature_importance = best_pipeline.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

# Save model
with open("fraud_model_advanced.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("Updated fraud model trained and saved successfully!")
