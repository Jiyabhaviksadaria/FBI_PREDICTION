import pandas as pd
import json

def process_data():
    print("Loading Train.xlsx...")
    df = pd.read_excel('Train.xlsx')
    
    # Clean data (remove NaNs in important columns if needed, though grouping handles a lot)
    if 'YEAR' in df.columns and 'MONTH' in df.columns and 'TYPE' in df.columns:
        print("Grouping data...")
        df_agg = df.groupby(['YEAR', 'MONTH', 'TYPE']).size().reset_index(name='Incident_Counts')
        
        # Merge with test shapes to see what we actually need to predict
        df_test = pd.read_csv('Test (2).csv')
        
        # Save processed data
        df_agg.to_csv('Processed_Crime_Data.csv', index=False)
        print("Saved Processed_Crime_Data.csv")
    else:
        print("Data columns:", list(df.columns))

if __name__ == '__main__':
    process_data()
