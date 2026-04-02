# 📊 ML Model Comparison Dashboard

## 🚀 Project Overview

This project is an interactive **Machine Learning Dashboard** developed using **Streamlit** as part of a lab assignment.
The main goal of this project is to allow users to **train and compare multiple machine learning models on a given dataset** through a simple and intuitive interface.

The dashboard enables users to upload datasets, select features and target variables, and visualize model performance using different evaluation metrics.

---

## 🎯 Objectives

* To understand and implement multiple classification algorithms
* To compare model performance using standard evaluation metrics
* To build an interactive ML application using Streamlit
* To visualize results for better interpretation

---

## ⚙️ Features Implemented

* 📂 Upload custom CSV datasets or use default dataset
* 🎯 Select target column dynamically (with validation)
* 🧠 Feature selection for training models
* 🔄 Automatic preprocessing:

  * Handling categorical data (encoding)
  * Handling missing values
* 🤖 Train multiple ML models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (SVM)
  * Naive Bayes

---

## 📊 Model Evaluation

The models are compared using:

* Accuracy
* Precision
* Recall
* Training Time

Additional visualizations:

* 🔍 Normalized Confusion Matrix
* 📈 ROC Curve (for binary classification)
* 📊 Performance comparison charts

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/ML-MODEL-COMPARISON-DASHBOARD.git
cd ML-Dashboard
```

2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

---

## 📌 Important Notes

* The target column should be **categorical** (e.g., Yes/No, 0/1)
* Avoid selecting ID or continuous numeric columns as target
* Feature selection helps improve performance and speed

---

## 📈 Conclusion

This project demonstrates how different machine learning models perform on the same dataset and highlights the importance of preprocessing, feature selection, and proper evaluation.

---

