import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading aggregated data...")
df_agg = pd.read_csv('Processed_Crime_Data.csv')

X_raw = df_agg[['YEAR', 'MONTH', 'TYPE']]
y = df_agg['Incident_Counts']

X = pd.get_dummies(X_raw, columns=['TYPE'], drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training baseline model...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Loading Test (2).csv...")
df_test_raw = pd.read_csv('Test (2).csv')
if 'Incident_Counts' in df_test_raw.columns:
    df_test_clean = df_test_raw.drop('Incident_Counts', axis=1)
else:
    df_test_clean = df_test_raw.copy()

X_test_final = pd.get_dummies(df_test_clean, columns=['TYPE'], drop_first=False)
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

print("Predicting...")
final_predictions = rf.predict(X_test_final)

df_test_raw['Incident_Counts'] = np.round(final_predictions).astype(int)
df_test_raw.to_csv('Final_FBI_Predictions_Jiya_Sadaria.csv', index=False)
print("Saved Final_FBI_Predictions_Jiya_Sadaria.csv")
