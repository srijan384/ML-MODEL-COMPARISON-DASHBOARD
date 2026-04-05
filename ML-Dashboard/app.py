import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

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

# Dataset size
st.write(f"📏 Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# -----------------------------
# Dataset Info
# -----------------------------
st.subheader("📌 Dataset Info")
st.write(df.describe())

# -----------------------------
# Feature Types
# -----------------------------
st.subheader("📊 Feature Types")

categorical_cols = df.select_dtypes(include=['object']).columns
numeric_cols = df.select_dtypes(exclude=['object']).columns

st.write("Categorical Columns:", list(categorical_cols))
st.write("Numeric Columns:", list(numeric_cols))

# Show category values
if len(categorical_cols) > 0:
    st.subheader("📂 Category Values")
    for col in categorical_cols:
        st.write(f"{col}:", df[col].unique())

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

# Data cleaning option
if st.checkbox("🧹 Clean Data (Drop Nulls)"):
    df = df.dropna()

# -----------------------------
# Select Target Column
# -----------------------------
target_column = st.selectbox("🎯 Select Target Column", df.columns)

# Validate target
if df[target_column].nunique() > 10:
    st.error("❌ Target column has too many unique values! Select categorical column.")
    st.stop()

# -----------------------------
# Feature Selection
# -----------------------------
features = st.multiselect(
    "🧠 Select Features",
    df.columns.drop(target_column),
    default=list(df.columns.drop(target_column))[:5]
)

# -----------------------------
# Data Preprocessing
# -----------------------------
if df[target_column].dtype == 'object':
    df[target_column] = df[target_column].astype('category').cat.codes

X = df[features]
y = df[target_column]

X = pd.get_dummies(X)
X = X.fillna(0)
y = y.fillna(0)

# -----------------------------
# Train Test Split
# -----------------------------
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
        with st.spinner("⏳ Training models..."):
            start_time = time.time()

            results, predictions = train_models(
                X_train, X_test, y_train, y_test, selected_models
            )

            total_time = time.time() - start_time

        # Add training time
        results["Training Time (s)"] = round(total_time, 3)

        # -----------------------------
        # Results
        # -----------------------------
        st.subheader("📊 Model Comparison Results")
        st.dataframe(results.style.highlight_max(axis=0))

        # Best model
        best_model = results.loc[results['Accuracy'].idxmax()]
        st.markdown(f"""
        ## 🏆 Best Model: {best_model['Model']}
        ### Accuracy: {best_model['Accuracy']:.2f}
        """)

        # Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

        # -----------------------------
        # Graphs
        # -----------------------------
        st.subheader("📈 Model Performance")
        st.bar_chart(results.set_index("Model")[["Accuracy", "Precision", "Recall"]])

        # -----------------------------
        # Confusion Matrix (Normalized)
        # -----------------------------
        st.subheader("🔍 Confusion Matrix")

        for model_name, y_pred in predictions.items():
            st.write(f"### {model_name}")

            cm = confusion_matrix(y_test, y_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, None]

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax)

            ax.set_title(f"{model_name} (Normalized)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)
            plt.close(fig)

        # -----------------------------
        # ROC Curve (Binary only)
        # -----------------------------
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ROC Curve (only for binary classification)
if len(y.unique()) == 2:

    st.subheader("📈 ROC Curve")

    fig, ax = plt.subplots()

    for model_name in selected_models:
        model = None

        # Recreate the same model
        if model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=200)

        elif model_name == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()

        elif model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()

        elif model_name == "KNN":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()

        elif model_name == "SVM":
            from sklearn.svm import SVC
            model = SVC(probability=True)  # IMPORTANT

        elif model_name == "Naive Bayes":
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()

        try:
            model.fit(X_train, y_train)

            # Get probabilities
            y_prob = model.predict_proba(X_test)[:, 1]

            # Compute ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

        except Exception as e:
            st.warning(f"{model_name} does not support ROC")

    # Random baseline
    ax.plot([0, 1], [0, 1], 'r--')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig)