# Streamlit AutoML with PyCaret

This Streamlit app allows users to upload a CSV file, select a target column, perform data cleaning, class balancing with SMOTE, scaling, and run AutoML using PyCaret to find the best classifier model.

## Features
- Upload CSV
- Handle missing values and duplicates
- SMOTE for class imbalance
- StandardScaler for feature scaling
- AutoML using PyCaret's compare_models
- Displays Accuracy, Classification Report, and Confusion Matrix

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
