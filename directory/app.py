# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gdown  # For downloading files from Google Drive
import os  # For handling file paths

# Set Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Google Drive file links
FILE_LINKS = {
    "X_train": "https://drive.google.com/uc?id=1iberDE0Lg9IZWMPCMmKXjJ4LgPokSyjm",
    "X_test": "https://drive.google.com/uc?id=1-mIMEAq4bOpLJNbt8DSfysMEvLyKSQKd",
    "y_test": "https://drive.google.com/uc?id=1UEhal_P-mHdfVCtqWJF07XBjHq9G5pca",
    "model": "https://drive.google.com/uc?id=12hER93fGFtl47CvX4SsPA-ECghT5kiZS"
}

# Function to download files
def download_file(url, output_path):
    if not os.path.exists(output_path):  # Download only if file doesn't already exist
        try:
            gdown.download(url, output_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download {output_path}. Error: {e}")

# Paths for local storage
LOCAL_FILES = {
    "X_train": "X_train_top.csv",
    "X_test": "X_test_top.csv",
    "y_test": "y_test.csv",
    "model": "lightgbm_fraud_model.pkl"
}

# Download datasets and model
for key, url in FILE_LINKS.items():
    download_file(url, LOCAL_FILES[key])

# Load datasets with error handling
try:
    X_train = pd.read_csv(LOCAL_FILES["X_train"])
    X_test = pd.read_csv(LOCAL_FILES["X_test"])
    y_test = pd.read_csv(LOCAL_FILES["y_test"])
except FileNotFoundError as e:
    st.error(f"Failed to load dataset. Error: {e}")
    st.stop()

# Try loading the LightGBM model with error handling
try:
    lgbm_model = joblib.load(LOCAL_FILES["model"])
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Add Fraud column for visualization
X_test['Fraud'] = y_test.values

# Map state numbers to names
state_mapping = {
    1: "Alabama", 2: "Alaska", 3: "Arizona", 4: "Arkansas", 5: "California", 6: "Colorado",
    7: "Connecticut", 8: "Delaware", 9: "Florida", 10: "Georgia", 11: "Hawaii", 12: "Idaho",
    13: "Illinois", 14: "Indiana", 15: "Iowa", 16: "Kansas", 17: "Kentucky", 18: "Louisiana",
    19: "Maine", 20: "Maryland", 21: "Massachusetts", 22: "Michigan", 23: "Minnesota",
    24: "Mississippi", 25: "Missouri", 26: "Montana", 27: "Nebraska", 28: "Nevada",
    29: "New Hampshire", 30: "New Jersey", 31: "New Mexico", 32: "New York", 33: "North Carolina",
    34: "North Dakota", 35: "Ohio", 36: "Oklahoma", 37: "Oregon", 38: "Pennsylvania",
    39: "Rhode Island", 40: "South Carolina", 41: "South Dakota", 42: "Tennessee", 43: "Texas",
    44: "Utah", 45: "Vermont", 46: "Virginia", 47: "Washington", 48: "West Virginia",
    49: "Wisconsin", 50: "Wyoming"
}

# Map states in X_test
X_test['StateName'] = X_test['State'].map(state_mapping)

# Fill missing state names with the most frequent state
most_frequent_state = X_test['StateName'].mode()[0]
X_test['StateName'] = X_test['StateName'].fillna(most_frequent_state)

# Function to predict fraud
def predict_fraud(data):
    try:
        predictions = lgbm_model.predict(data)
        probabilities = lgbm_model.predict_proba(data)[:, 1]  # Probability of fraud
        return predictions, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Main Streamlit Application
def main():
    st.title("📊 Healthcare Provider Fraud Detection Dashboard")
    st.markdown("""
        ## About This Application
        This application is designed to detect potential fraud among healthcare providers based on their claims data.
        It allows users to:
        - Predict whether a selected healthcare provider is involved in fraudulent activities.
        - Explore insights and patterns in the dataset through visualizations.
        - Understand the underlying factors contributing to fraud detection.

        The predictions are powered by a pre-trained **LightGBM model** built on a structured dataset, ensuring accuracy and reliability.
    """)

    # Tabs for different sections
    tabs = st.tabs(["Dataset Summary", "Insights", "Predictions"])

    # Tab 1: Dataset Summary
    with tabs[0]:
        st.header("📂 Dataset Summary")
        st.write(f"### Dataset Sizes:")
        st.write(f"- **Training Dataset:** {X_train.shape[0]} rows, {X_train.shape[1]} columns")
        st.write(f"- **Testing Dataset:** {X_test.shape[0]} rows, {X_test.shape[1]} columns")
        st.write("### Sample Data from Testing Dataset:")
        st.write(X_test.head())

    # Tab 2: Insights
    with tabs[1]:
        st.header("📈 Insights and Visualizations")

        # Fraud vs. Non-Fraud counts
        st.subheader("1️⃣ Fraud vs. Non-Fraud Counts")
        plt.figure(figsize=(3, 2))
        sns.countplot(x='Fraud', data=X_test, palette='Set2')
        plt.title("Fraud vs. Non-Fraud Counts", fontsize=9)
        plt.xlabel("Fraud (0: Non-Fraud, 1: Fraud)", fontsize=8)
        plt.ylabel("Count", fontsize=8)
        st.pyplot(plt)

        # State-wise fraud distribution
        st.subheader("2️⃣ State-wise Fraud Distribution")
        state_fraud = X_test.groupby('StateName')['Fraud'].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 4))
        state_fraud.plot(kind='bar', color='teal')
        plt.title("State-wise Fraud Distribution", fontsize=12)
        plt.xlabel("State", fontsize=10)
        plt.ylabel("Average Fraud Rate", fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)

        # Correlation heatmap
        st.subheader("3️⃣ Correlation Heatmap")
        corr_matrix = X_train.corr()  # Compute the correlation matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title("Correlation Heatmap of Selected Features", fontsize=12)
        st.pyplot(plt)

    # Tab 3: Predictions
    with tabs[2]:
        st.header("🔍 Predictions")
        provider_id = st.selectbox("Select a Provider from Testing Dataset", X_test.index)
        provider_data = X_test.iloc[[provider_id]].drop(columns=['Fraud', 'StateName'])

        st.write("### Provider Details:")
        st.write(provider_data)

        if st.button("Predict Fraud"):
            prediction, probability = predict_fraud(provider_data)
            if prediction is not None:
                fraud_probability = probability[0] * 100
                provider_name = X_test.iloc[provider_id]['Provider']
                if prediction[0] == 1:
                    st.success(f"🚩 The provider **{provider_name}** is **FRAUD** with a probability of **{fraud_probability:.2f}%**.")
                else:
                    st.success(f"✅ The provider **{provider_name}** is **NOT FRAUD** with a probability of **{fraud_probability:.2f}%**.")

if __name__ == "__main__":
    main()
