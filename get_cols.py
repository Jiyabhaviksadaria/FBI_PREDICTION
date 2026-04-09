import pandas as pd
import json
df = pd.read_excel('Train.xlsx', nrows=5)
print(df.columns.tolist())
