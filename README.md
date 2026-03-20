# 🤖 ML Model Comparison Dashboard

A colorful and interactive **Streamlit** application for uploading datasets, training multiple classic ML models, and comparing their performance side-by-side with rich visualizations.

---

## ✨ Features

| Feature | Details |
|---|---|
| **File Upload** | CSV and Excel (`.xlsx` / `.xls`) |
| **Data Exploration** | Statistics, data types, missing values, correlation heatmap |
| **7 Classic ML Models** | Logistic Regression, Random Forest, SVM, Gradient Boosting, KNN, Naive Bayes, Decision Tree |
| **Metrics** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Visualizations** | Grouped bar chart, radar chart, ROC curves, confusion matrices, feature importance |
| **Export** | Download metrics as CSV or a full text report |

---

## 🚀 Quick Start

### 1 · Clone the repo
```bash
git clone https://github.com/avneetkaur3513/ML-Model-Comparison-Dashboard.git
cd ML-Model-Comparison-Dashboard
```

### 2 · Install dependencies
```bash
pip install -r requirements.txt
```

### 3 · Run the app
```bash
streamlit run app.py
```

The dashboard opens automatically in your browser at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
ML-Model-Comparison-Dashboard/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 📖 Usage Guide

1. **Upload** a CSV or Excel file from the sidebar.
2. **Select** feature columns and the target column.
3. Adjust the **train/test split** ratio.
4. Choose which **models** to train.
5. Click **🚀 Train Models**.
6. Explore results across four tabs:
   - **Data Exploration** – dataset preview, statistics, distributions
   - **Model Results** – leaderboard and classification reports
   - **Visualizations** – charts and confusion matrices
   - **Export** – download results

---

## 🔧 Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

---

## 📄 License

MIT
