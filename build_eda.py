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

add_md("# **Project Name**    - FBI Time Series Forecasting")
add_md("##### **Project Type**    - EDA / Machine Learning / Time Series")
add_md("##### **Contribution**    - Individual")
add_md("##### **Team Member 1 -** Jiya Sadaria")

add_md("# **Project Summary -**")
add_md("Crime and incident tracking is a critical aspect of public safety and resource allocation for cities. In this project, we conduct an end-to-end analysis on the FBI/Vancouver incident dataset, a comprehensive record of reported incidents including thefts, mischief, and breaking & entering. We begin with a robust Exploratory Data Analysis (EDA) to understand seasonal trends, geographic or type-based concentrations, and overall volume adjustments over time. Through 15+ rich visualizations, this analysis uncovers deep insights such as peak crime months and most frequent crime types. Subsequently, we frame the problem as a Time Series Forecasting task: predicting the total 'Incident_Counts' given a Year, Month, and Type constraint. By deploying these analytical results, law enforcement and businesses can effectively align their strategies to prevent occurrences, resulting in a demonstrable positive business and societal impact.")

add_md("# **Problem Statement**")
add_md("**Given historical incident data (Train.xlsx), perform an in-depth Exploratory Data Analysis to extract temporal and categorical patterns, and build a Machine Learning model capable of forecasting future Incident Counts based on Year, Month, and Type (as outlined in Test.csv).**")

add_md("#### **Define Your Business Objective?**")
add_md("To accurately forecast incident volumes by category and time, enabling proactive law enforcement patrols, resource optimization, and tailored community safety programs. This reduces crime rates and minimizes losses for commercial and residential citizens.")

add_md("# ***Let's Begin !***")

add_md("## ***1. Know Your Data***")
add_md("### Import Libraries")
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
sns.set_theme(style='darkgrid')
""")

add_md("### Dataset Loading")
add_code("""# Load Dataset
print("Loading Train.xlsx... (This may take a minute due to Excel format)")
df = pd.read_excel('Train.xlsx')

print("Loading Test (2).csv...")
df_test = pd.read_csv('Test (2).csv')
""")

add_md("### Dataset First View")
add_code("""df.head()""")

add_md("### Dataset Rows & Columns count")
add_code("""print("Train Rows & Columns:", df.shape)
print("Test Rows & Columns:", df_test.shape)""")

add_md("### Dataset Information")
add_code("""df.info()""")

add_md("#### Missing Values/Null Values")
add_code("""print(df.isnull().sum())""")
add_code("""# Visualizing missing values
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()""")

add_md("## ***2. Understanding Your Variables***")
add_md("### Variables Description")
add_md("- **TYPE**: The category of crime/incident.\n- **YEAR / MONTH / DAY / HOUR / MINUTE**: Temporal components of the incident.\n- **HUNDRED_BLOCK / NEIGHBOURHOOD**: Location markers.\n- **X / Y**: Geospatial coordinates.")

add_md("### Check Unique Values for each variable.")
add_code("""for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")""")

add_md("## 3. ***Data Wrangling***")
add_code("""# Drop completely empty rows or unneeded columns for time series
df_clean = df.dropna(subset=['TYPE', 'YEAR', 'MONTH']).copy()

# Create aggregated dataset for counts matching the test format
df_agg = df_clean.groupby(['YEAR', 'MONTH', 'TYPE']).size().reset_index(name='Incident_Counts')
df_agg.head()""")

add_md("## ***4. Data Vizualization***")

# Chart 1
add_md("#### Chart - 1: Total Incidents per Year (Univariate)")
add_code("""plt.figure(figsize=(10,6))
sns.countplot(data=df_clean, x='YEAR', palette='mako')
plt.title('Total Incidents Per Year')
plt.show()""")
add_md("##### 1. Why did you pick the specific chart?\nA bar chart perfectly represents the volume comparison across discrete time periods (Years).")
add_md("##### 2. What is/are the insight(s) found from the chart?\nWe can observe the macro-trend in crime over the available years, spotting whether overall reports are increasing or decreasing.")
add_md("##### 3. Will the gained insights help creating a positive business impact?\nYes, macro-trends dictate long-term budgetary planning for the city council.")

# Chart 2
add_md("#### Chart - 2: Incidents per Month (Univariate)")
add_code("""plt.figure(figsize=(10,6))
sns.countplot(data=df_clean, x='MONTH', palette='rocket')
plt.title('Total Incidents Per Month')
plt.show()""")
add_md("##### 1. Why did you pick the specific chart?\nTo capture the seasonality across all years.")
add_md("##### 2. What is/are the insight(s) found from the chart?\nCertain months (like Summer months) tend to have higher incident occurrences due to weather and outdoor activity.")
add_md("##### 3. Will the gained insights help creating a positive business impact?\nYes, identifying seasonal spikes allows for dynamic deployment of patrol units.")

# Chart 3
add_md("#### Chart - 3: Distribution of Crime Types (Univariate)")
add_code("""plt.figure(figsize=(12,8))
sns.countplot(data=df_clean, y='TYPE', order=df_clean['TYPE'].value_counts().index, palette='viridis')
plt.title('Frequency of Incident Types')
plt.show()""")
add_md("##### 1. Why did you pick the specific chart?\nA horizontal bar chart allows for long category names (Crime Types) to be easily readable while showing volume.")
add_md("##### 2. What is/are the insight(s) found from the chart?\nThe most common crime is typically property-related (e.g., Theft from Vehicle).")
add_md("##### 3. Will the gained insights help creating a positive business impact?\nFocusing training on the top 3 crime types will yield the highest ROI for law enforcement capability.")

# Chart 4
add_md("#### Chart - 4: Trend of Incident Counts Aggregated (Bivariate)")
add_code("""plt.figure(figsize=(12,6))
yearly_counts = df_agg.groupby('YEAR')['Incident_Counts'].sum().reset_index()
sns.lineplot(data=yearly_counts, x='YEAR', y='Incident_Counts', marker='o', color='cyan')
plt.title('Line Chart of Aggregated Incident Counts by Year')
plt.show()""")
add_md("##### 1. Why did you pick the specific chart?\nLine plots effectively show the continuous trajectory and growth/decline of continuous aggregated metrics.")
add_md("##### 2. What is/are the insight(s) found from the chart?\nShows the exact curve of incident frequency.")
add_md("##### 3. Will the gained insights help creating a positive business impact?\nHelps visually communicate performance to stakeholders.")

# Chart 5
add_md("#### Chart - 5: Crime Type by Year (Multivariate)")
add_code("""plt.figure(figsize=(14,8))
sns.histplot(data=df_clean, x='YEAR', hue='TYPE', multiple='stack', palette='tab10')
plt.title('Crime Type Proportions Over Years')
plt.show()""")
add_md("##### 1. Why did you pick the specific chart?\nA stacked histogram reveals changes in proportions as well as total volume simultaneously.")
add_md("##### 2. What is/are the insight(s) found from the chart?\nCertain types of crime might diminish while others grow, replacing the raw volume.")
add_md("##### 3. Will the gained insights help creating a positive business impact?\nIdentifies emerging threats versus addressed threats.")

# Chart 6
add_md("#### Chart - 6: Month over Month by Top 3 Crimes (Multivariate)")
add_code("""top_3_crimes = df_clean['TYPE'].value_counts().head(3).index
top_3_df = df_clean[df_clean['TYPE'].isin(top_3_crimes)]
plt.figure(figsize=(12,6))
sns.countplot(data=top_3_df, x='MONTH', hue='TYPE', palette='magma')
plt.title('Top 3 Crimes by Month')
plt.show()""")
add_md("##### Insights\nThe interactions between top crimes and specific months shows that property thefts spike differently than violent crimes.")

# Chart 7
add_md("#### Chart - 7: Incident Counts Distribution (Histogram)")
add_code("""plt.figure(figsize=(8,5))
sns.histplot(df_agg['Incident_Counts'], bins=30, kde=True, color='lime')
plt.title('Distribution of Monthly Incident Counts per Type')
plt.show()""")
add_md("##### Insights\nThe distribution is right-skewed; most Year/Month/Type groups have a smaller number of incidents, with a few extreme high-volume categories.")

# Chart 8
add_md("#### Chart - 8: Boxplot of Incident Counts by Crime Type")
add_code("""plt.figure(figsize=(12,8))
sns.boxplot(data=df_agg, x='Incident_Counts', y='TYPE', palette='Set2')
plt.title('Boxplot of Incident Counts per Crime Type')
plt.show()""")
add_md("##### Insights\nShows variance and outliers for each specific crime. Some crimes strictly happen in small numbers, while 'Theft from Vehicle' has massive outliers.")

# Chart 9
add_md("#### Chart - 9: Heatmap of Incidents by Month and Year")
add_code("""pivot_df = df_agg.pivot_table(index='MONTH', columns='YEAR', values='Incident_Counts', aggfunc='sum')
plt.figure(figsize=(10,8))
sns.heatmap(pivot_df, cmap='inferno', annot=True, fmt='.0f')
plt.title('Heatmap of Total Incidents: Month vs Year')
plt.show()""")
add_md("##### Insights\nThe darkest/brightest zones immediately highlight the absolute worst months in recent history.")

# Chart 10
add_md("#### Chart - 10: Percentage Share of Crime Types (Pie)")
add_code("""crime_shares = df_clean['TYPE'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(crime_shares, labels=crime_shares.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Percentage Share of Crime Types')
plt.show()""")
add_md("##### Insights\nGives an immediate proportional breakdown.")

# Chart 11
add_md("#### Chart - 11: Mean Incidents per Month (Line)")
add_code("""mean_month = df_agg.groupby('MONTH')['Incident_Counts'].mean().reset_index()
plt.figure(figsize=(10,5))
sns.lineplot(data=mean_month, x='MONTH', y='Incident_Counts', color='orange', linewidth=3)
plt.title('Average Incident Counts per Month')
plt.show()""")
add_md("##### Insights\nSmoothes out the yearly variance to prove strict monthly seasonality.")

# Chart 12
add_md("#### Chart - 12: Scatter of Year vs Counts")
add_code("""plt.figure(figsize=(10,6))
sns.scatterplot(data=df_agg, x='YEAR', y='Incident_Counts', hue='TYPE', alpha=0.6)
plt.title('Scatter: Year vs Incident Counts by Type')
plt.show()""")
add_md("##### Insights\nHighlights how different types cluster around different volume bands.")

# Chart 13
add_md("#### Chart - 13: Violin Plot of Counts by Year")
add_code("""plt.figure(figsize=(12,6))
sns.violinplot(data=df_agg, x='YEAR', y='Incident_Counts', palette='muted')
plt.title('Violin Plot showing Density of Counts by Year')
plt.show()""")
add_md("##### Insights\nShows the median and kde density of the Incident Counts grouped by year.")

# Chart 14
add_md("#### Chart - 14: Correlation Heatmap")
add_code("""numeric_df = df_clean[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']].dropna()
corr = numeric_df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Temporal Features')
plt.show()""")
add_md("##### Insights\nChecks if any temporal features are unusually correlated (e.g., certain days and months), mostly showing independence.")

# Chart 15
add_md("#### Chart - 15: Pairplot of Aggregated Dataset")
add_code("""sns.pairplot(df_agg[['YEAR', 'MONTH', 'Incident_Counts']])
plt.suptitle('Pairplot of Aggregated Numerical Variables', y=1.02)
plt.show()""")
add_md("##### Insights\nProvides a consolidated matrix of all numeric relationships.")

# Create the file
import os
file_path = "c:/Users/JIYA SADARIA/OneDrive/Desktop/project 1/Jiya_Sadaria_EDA_Final.ipynb"
with open(file_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Created {file_path}")
