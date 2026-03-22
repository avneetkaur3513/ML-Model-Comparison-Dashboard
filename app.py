import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
    }
    h1, h2, h3, h4, h5, h6 { color: #f0f0f0 !important; }
    p, li, span, label { color: #d0d0e0 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border-radius: 25px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

COLORS = ["#667eea", "#f5576c", "#4facfe", "#43e97b", "#fa709a", "#fee140", "#30cfd0", "#a18cd1"]
PLOTLY_TEMPLATE = "plotly_dark"

# ════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(file_input):
    """Handles both uploaded files and URLs."""
    try:
        if isinstance(file_input, str): # If it's the GitHub URL
            return pd.read_csv(file_input)
        
        name = file_input.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file_input)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_input)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df, feature_cols, target_col):
    subset = df[feature_cols + [target_col]].dropna()
    X = subset[feature_cols].copy()
    y = subset[target_col].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == "object" or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.values
    return X.values, y

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    }

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metric_row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # ROC-AUC is optional because not all datasets/models can compute it.
        try:
            if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                metric_row["ROC-AUC"] = roc_auc_score(y_test, y_prob)
            elif len(np.unique(y_test)) > 2 and hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                metric_row["ROC-AUC"] = roc_auc_score(y_test, y_prob, multi_class="ovr")
            else:
                metric_row["ROC-AUC"] = np.nan
        except Exception:
            metric_row["ROC-AUC"] = np.nan

        results.append(metric_row)
        reports[name] = classification_report(y_test, y_pred, zero_division=0)

    return pd.DataFrame(results).sort_values("F1", ascending=False), reports

# (Note: Confusion Matrix and other plot functions from your original code are assumed to be here)
# Copy-paste your plot_metric_comparison, plot_radar, etc. here...

def plot_data_distribution(df, target_col):
    value_counts = df[target_col].value_counts().reset_index()
    value_counts.columns = [target_col, "Count"]
    fig = px.pie(value_counts, names=target_col, values="Count", title=f"Target Distribution", template=PLOTLY_TEMPLATE, hole=0.4)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    return fig

def plot_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", title="Correlation Heatmap", template=PLOTLY_TEMPLATE)
    return fig

# ════════════════════════════════════════════════════════════════════════════
# Main app
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style="text-align:center; padding:20px 0;">
        <h1 style="font-size:2.5rem; font-weight:800; background:linear-gradient(90deg,#667eea,#f5576c,#43e97b); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🤖 ML Model Comparison Dashboard
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # Use RAW URL for GitHub data
    DEFAULT_DATA_URL = "https://raw.githubusercontent.com/avneetkaur3513/ML-Model-Comparison-Dashboard/main/sample_data.csv"

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("✅ Using uploaded file")
        else:
            df = load_data(DEFAULT_DATA_URL)
            st.info("💡 Using GitHub sample data")

        if df is not None:
            all_cols = df.columns.tolist()
            target_col = st.selectbox("🎯 Target Column", all_cols, index=len(all_cols) - 1)
            feature_cols = st.multiselect("📋 Feature Columns", [c for c in all_cols if c != target_col], default=[c for c in all_cols if c != target_col])
            test_size = st.slider("🔀 Test Size", 0.1, 0.5, 0.2)
            
            all_model_list = list(get_models().keys())
            selected_model_names = st.multiselect("🧠 Models", all_model_list, default=all_model_list[:3])
            train_btn = st.button("🚀 Train Models")
        else:
            st.stop()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_explore, tab_results = st.tabs(["🔍 Explore", "📊 Results"])

    with tab_explore:
        st.dataframe(df.head(50))
        st.plotly_chart(plot_data_distribution(df, target_col))

    with tab_results:
        if train_btn:
            if not feature_cols:
                st.error("Please select at least one feature column.")
                st.stop()
            if not selected_model_names:
                st.error("Please select at least one model.")
                st.stop()

            X, y = preprocess_data(df, feature_cols, target_col)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            selected_models = {name: get_models()[name] for name in selected_model_names}

            with st.spinner("Training in progress..."):
                results_df, reports = evaluate_models(selected_models, X_train, X_test, y_train, y_test)

            st.success("Training complete.")
            st.subheader("Model Performance")
            st.dataframe(results_df, use_container_width=True)

            plot_df = results_df.melt(
                id_vars="Model",
                value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                var_name="Metric",
                value_name="Score",
            )
            fig = px.bar(
                plot_df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=COLORS,
                title="Metric Comparison by Model",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Classification Reports")
            for model_name, report_text in reports.items():
                with st.expander(model_name):
                    st.text(report_text)
        else:
            st.info("Click 'Train Models' to see results.")

if __name__ == "__main__":
    main()
