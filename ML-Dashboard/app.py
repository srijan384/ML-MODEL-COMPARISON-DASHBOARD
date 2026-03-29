import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from model import train_models

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 ML Model Comparison Dashboard")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

st.subheader("📂 Dataset Preview")
st.dataframe(df)

# -----------------------------
# Dataset Info
# -----------------------------
st.subheader("📌 Dataset Info")
st.write(df.describe())

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

selected_models = st.sidebar.multiselect(
    "Select Models",
    ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM", "Naive Bayes"],
    default=["Logistic Regression", "Decision Tree"]
)

# Feature Selection
features = st.sidebar.multiselect(
    "Select Features",
    df.columns[:-1],
    default=list(df.columns[:-1])
)

# -----------------------------
# Prepare Data
# -----------------------------
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------
if st.button("🚀 Train Models"):

    if len(selected_models) == 0:
        st.warning("Please select at least one model!")
    else:
        results, predictions = train_models(
            X_train, X_test, y_train, y_test, selected_models
        )

        # -----------------------------
        # Results Table
        # -----------------------------
        st.subheader("📊 Model Comparison Results")
        st.dataframe(results.style.highlight_max(axis=0))

        # -----------------------------
        # Best Model
        # -----------------------------
        best_model = results.loc[results['Accuracy'].idxmax()]
        st.success(f"🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.2f})")

        # -----------------------------
        # Graphs
        # -----------------------------
        st.subheader("📈 Overall Comparison")
        st.bar_chart(results.set_index("Model"))

        st.subheader("📊 Accuracy Comparison")
        st.bar_chart(results.set_index("Model")["Accuracy"])

        st.subheader("📊 Precision Comparison")
        st.bar_chart(results.set_index("Model")["Precision"])

        st.subheader("📊 Recall Comparison")
        st.bar_chart(results.set_index("Model")["Recall"])

        st.subheader("📈 Metric Trends")
        st.line_chart(results.set_index("Model"))

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("🔍 Confusion Matrix")

        for model_name, y_pred in predictions.items():
            st.write(f"### {model_name}")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)