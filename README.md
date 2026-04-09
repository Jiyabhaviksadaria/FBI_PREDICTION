# 🕵️‍♀️ FBI Crime Time-Series Forecasting

## 📌 Project Overview
This project focuses on performing an end-to-end data science lifecycle to forecast FBI crime statistics using time-series data. It involves comprehensive Exploratory Data Analysis (EDA), extensive data preprocessing, and the training of a predictive regression model to generate future crime predictions.

## 📂 Repository Structure

The project is organized into structured modular utility scripts and comprehensive Jupyter Notebooks for analysis and training:

- **`Jiya_Sadaria_EDA_Final.ipynb`**: Comprehensive Exploratory Data Analysis (EDA) containing 15+ insightful visualizations, deep-dive data profiling, and statistical summaries using the UBM (Univariate, Bivariate, Multivariate) rule.
- **`Jiya_Sadaria_ML_Final.ipynb`**: The Machine Learning pipeline. Contains data preprocessing, baseline model building, hyperparameter tuning using a Random Forest Regressor, and final model evaluation.
- **`build_eda.py` | `build_ml.py`**: Python scripts used to assist in the building and generation of the EDA and ML pipelines.
- **`preprocess_data.py` | `get_cols.py` | `inspect_train.py`**: Utility scripts to perform modularized data profiling, column extraction, and robust cleaning strategies.
- **`generate_final_csv.py`**: Automates the generation of the final submission file containing the time-series forecasting predictions.
- **`Final_FBI_Predictions_Jiya_Sadaria.csv`**: The end-result predictions generated from the final inference pipeline.

> **Note:** Large training and testing datasets (such as `Train.xlsx` and `Test (2).csv`) have been intentionally omitted from this repository via `.gitignore` to keep the codebase perfectly clean and strictly focused on source code.

## 🚀 Methodology

1. **Data Cleaning & Wrangling**: Processed raw demographic and crime reporting data, handling missing values and ensuring consistency.
2. **Exploratory Data Analysis (EDA)**: Visualized complex crime trends over time. Examined the correlation between various jurisdiction variables and actual crimes reported.
3. **Machine Learning Pipeline**: 
   - Transformed the given tabular dataset into a supervised format suitable for forecasting.
   - Designed a robust machine learning inference pipeline.
   - Trained an optimized **Random Forest Regressor** to predict crime estimates.
   - Evaluated models using rigorous cross-validation to minimize variance and Mean Absolute Error (MAE).
4. **Final Inference Pipeline**: Dynamically built constraints around validation layers to gracefully format output predictions into expected structures.

## 🛠️ Setup & Installation

To run this project locally, simply clone the repository and set up a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/Jiyabhaviksadaria/FBI_PREDICTION.git
cd FBI_PREDICTION

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install required numerical and data manipulation dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter openpyxl
```

## 👩‍💻 Usage

The easiest way to explore this project is interactively via Jupyter Notebooks:

```bash
jupyter notebook
```
- Dive into `Jiya_Sadaria_EDA_Final.ipynb` to read through the analytical narrative.
- Open `Jiya_Sadaria_ML_Final.ipynb` to view the finalized model hyperparameters, performance evaluations, and prediction mapping.

If you are evaluating programmatic data transformation or building inference wrappers, you can simply trace the logic in standard pure Python via:
```bash
python generate_final_csv.py
```

## 📜 Author
**Jiya Sadaria**
B.Tech AIML Student | Focus on Predictive Models and Applied Machine Learning