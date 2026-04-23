import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page Config
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Title
st.title("🎓 Student Performance Prediction System")
st.write("A Machine Learning project for predicting student results")

# Sidebar Navigation
menu = st.sidebar.selectbox(" Menu", ["Home", "Upload Data", "Train Model", "Predict"])

# Session state for model
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.features = None
    st.session_state.df = None


# ---------------- HOME ----------------
if menu == "Home":
    st.subheader(" Project Overview")
    st.write("""
    This project predicts student performance using Machine Learning.
    
    🔹 Upload dataset  
    🔹 Train ML model  
    🔹 Predict results  
    
    Built using:
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    """)


# ---------------- UPLOAD DATA ----------------
elif menu == "Upload Data":
    uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.success(" Data Loaded Successfully")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Statistics")
        st.write(df.describe())

        # Visualization
        st.subheader("Correlation Heatmap")

        
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr()




        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))

        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

        st.pyplot(fig)


# ---------------- TRAIN MODEL ----------------
elif menu == "Train Model":
    df = st.session_state.df

    if df is None:
        st.warning(" Please upload dataset first")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.subheader(" Model Configuration")

        target = st.selectbox("Select Target", numeric_cols)
        features = st.multiselect(
            "Select Features",
            [col for col in numeric_cols if col != target]
        )

        if len(features) == 0:
            st.warning("Select at least one feature")
        else:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            st.session_state.model = model
            st.session_state.features = features

            st.success(f"Model Trained | R² Score: {score:.2f}")


# ---------------- PREDICT ----------------
elif menu == "Predict":
    model = st.session_state.model
    features = st.session_state.features
    df = st.session_state.df

    if model is None:
        st.warning(" Train model first")
    else:
        st.subheader(" Make Prediction")

        input_data = {}

        for feature in features:
            input_data[feature] = st.number_input(
                f"{feature}",
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].mean())
            )

        if st.button(" Predict"):
            input_df = pd.DataFrame([input_data])
            result = model.predict(input_df)[0]
            st.success(f" Predicted Score: {result:.2f}")