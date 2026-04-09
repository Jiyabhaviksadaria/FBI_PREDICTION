import json

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {"private_outputs": True},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"}
  },
  "cells": []
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text + "\n"]
    })

def add_code(code):
    lines = [line + "\n" if i < len(code.split('\n')) - 1 else line for i, line in enumerate(code.split('\n'))]
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "execution_count": None,
        "outputs": []
    })

add_md("# **Project Name**    - FBI Time Series Forecasting (Machine Learning)")
add_md("##### **Project Type**    - Regression / Time Series")
add_md("##### **Contribution**    - Individual")
add_md("##### **Team Member 1 -** Jiya Sadaria")

add_md("# **Project Summary -**")
add_md("Following our comprehensive Exploratory Data Analysis, this notebook transitions into predictive modeling. Our goal is to train a robust Machine Learning algorithm that can ingest temporal features (Year, Month) and categorical features (Crime Type) to accurately forecast the Expected Incident Counts. We carefully prepare our aggregated dataset, separating it into training and evaluation sets. We utilize an ensemble learning approach (Random Forest Regressor) and fine-tune its hyper-parameters via GridSearchCV to optimize results, ensuring we capture complex non-linear trends inherent in seasonal crime data. We finally run predictions on the unseen Test.csv and output our final forecasts.")

add_md("## ***1. Know Your Data & Data Wrangling***")
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_theme(style='darkgrid')
""")

add_code("""print("Loading aggregated data directly leveraging Processed_Crime_Data.csv (Created in EDA Phase)")
# Since we need to join this pipeline, let's recreate the aggregation if it doesn't exist, or load it.
try:
    df_agg = pd.read_csv('Processed_Crime_Data.csv')
    print("Found existing processed data.")
except FileNotFoundError:
    print("Reprocessing Train.xlsx...")
    df = pd.read_excel('Train.xlsx')
    df_clean = df.dropna(subset=['TYPE', 'YEAR', 'MONTH'])
    df_agg = df_clean.groupby(['YEAR', 'MONTH', 'TYPE']).size().reset_index(name='Incident_Counts')

df_agg.head()
""")

add_md("## ***2. Feature Engineering & Preprocessing***")
add_md("We must encode the categorical `TYPE` mathematically so the Random Forest model can interpret it via One-Hot Encoding.")
add_code("""# Setup X and y
X_raw = df_agg[['YEAR', 'MONTH', 'TYPE']]
y = df_agg['Incident_Counts']

# One-Hot Encoding
X = pd.get_dummies(X_raw, columns=['TYPE'], drop_first=False)
X.head()""")

add_md("## ***3. Train-Test Split***")
add_code("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])""")

add_md("## ***4. Machine Learning Model Setup***")
add_md("### Training a Random Forest Regressor")
add_code("""# Initialize baseline model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict on test
y_pred = rf.predict(X_test)
""")

add_md("### Evaluation Metrics")
add_code("""rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Baseline Random Forest RMSE: {rmse:.2f}")
print(f"Baseline Random Forest MAE: {mae:.2f}")
print(f"Baseline Random Forest R^2 Score: {r2:.2f}")
""")
add_md("##### Business Impact of Metrics:\nThe **MAE (Mean Absolute Error)** directly tells law enforcement how far off their expected incident count will be on average. A low MAE ensures efficient patrol deployment without over/under-staffing.")

add_md("## ***5. Cross Validation & Hyperparameter Tuning***")
add_md("We apply `GridSearchCV` to optimize the Random Forest")
add_code("""# Define parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

print("Running GridSearchCV (this may take a moment)...")
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

print("Best Parameters found: ", grid_search.best_params_)
best_rf = grid_search.best_estimator_

# Re-evaluate
y_pred_tuned = best_rf.predict(X_test)
tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
tuned_r2 = r2_score(y_test, y_pred_tuned)

print(f"Tuned Random Forest RMSE: {tuned_rmse:.2f}")
print(f"Tuned Random Forest R^2 Score: {tuned_r2:.2f}")
""")

add_md("### Feature Importance")
add_code("""# Display feature importances
importances = best_rf.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', palette='mako')
plt.title('Top 10 Most Important Features')
plt.show()""")

add_md("## ***6. Generate Final Test Predictions***")
add_md("Now, we load the `Test (2).csv` which asks us to predict for specific Years, Months, and Types.")
add_code("""# Load original test file
df_test_raw = pd.read_csv('Test (2).csv')

# Drop the Incident_Counts column if it is empty/NaN so we can replace it
if 'Incident_Counts' in df_test_raw.columns:
    df_test_clean = df_test_raw.drop('Incident_Counts', axis=1)
else:
    df_test_clean = df_test_raw.copy()

# Apply the EXACT same One-Hot Encoding format. We use `reindex` to ensure columns perfectly match the training set X.
X_test_final = pd.get_dummies(df_test_clean, columns=['TYPE'], drop_first=False)
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

# Predict using the Tuned Model
final_predictions = best_rf.predict(X_test_final)

# Append predictions safely
df_test_raw['Incident_Counts'] = np.round(final_predictions).astype(int)

# Preview final output
print(df_test_raw.head())
""")

add_code("""# EXPORT CSV
filename = 'Final_FBI_Predictions_Jiya_Sadaria.csv'
df_test_raw.to_csv(filename, index=False)
print(f"Successfully saved test predictions to {filename}!")
""")

# Create the file
import os
file_path = "c:/Users/JIYA SADARIA/OneDrive/Desktop/project 1/Jiya_Sadaria_ML_Final.ipynb"
with open(file_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Created {file_path}")
