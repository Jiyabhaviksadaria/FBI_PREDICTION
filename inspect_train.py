import pandas as pd, json, sys
path = r"C:/Users/JIYA SADARIA/OneDrive/Desktop/project 1/Train.xlsx"
df = pd.read_excel(path)
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print(df.head().to_json(orient='records'))
