import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import setup, compare_models, predict_model, pull

st.set_page_config(page_title="AutoML with PyCaret", layout="wide")
st.title("🤖 AutoML Pipeline using PyCaret")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Data Preview", df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns)
    if target_col:
        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X = pd.get_dummies(X)
        if len(np.unique(y)) == 2:
            sm = SMOTE(random_state=42)
            X_resampled, y_resampled = sm.fit_resample(X, y)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_resampled)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            full_df = X_scaled_df.copy()
            full_df[target_col] = y_resampled.values

            st.info("⚙️ Setting up PyCaret environment...")
            s = setup(data=full_df, target=target_col, silent=True, session_id=123, verbose=False)
            best_model = compare_models()

            st.success("✅ AutoML Completed")
            st.write("### 🏆 Best Model Leaderboard")
            st.dataframe(pull())

            st.write("### 📈 Evaluation Metrics on Test Set")
            predictions = predict_model(best_model)
            y_pred = predictions["Label"]
            y_true = predictions[target_col]

            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            matrix = confusion_matrix(y_true, y_pred)

            st.metric("🎯 Accuracy", f"{acc:.4f}")

            st.write("📋 Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            st.write("🧮 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Only binary classification supported currently.")
