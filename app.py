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
    /* Main background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
        color: white;
        margin-bottom: 10px;
    }
    .metric-card h3 { font-size: 2rem; font-weight: 700; margin: 5px 0; }
    .metric-card p  { font-size: 0.9rem; opacity: 0.85; margin: 0; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Streamlit elements */
    .stMetric { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 12px; }
    div[data-testid="stMetricValue"] { color: #f093fb !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #a0a0c0 !important; }

    h1, h2, h3, h4, h5, h6 { color: #f0f0f0 !important; }
    p, li, span, label { color: #d0d0e0 !important; }

    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(102,126,234,0.5); }

    .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { color: #a0a0c0 !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg,#667eea,#764ba2); border-radius: 10px; color: white !important; }

    .stDataFrame { background: rgba(255,255,255,0.03); }
    .stSelectbox > div, .stMultiSelect > div { background: rgba(255,255,255,0.05) !important; }

    div.stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette shared across charts ──────────────────────────────────────
COLORS = [
    "#667eea", "#f5576c", "#4facfe", "#43e97b",
    "#fa709a", "#fee140", "#30cfd0", "#a18cd1",
]
PLOTLY_TEMPLATE = "plotly_dark"


# ════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload CSV or Excel.")
        return None


def preprocess_data(df, feature_cols, target_col):
    """Encode categoricals, drop NaN rows, return X and y arrays."""
    subset = df[feature_cols + [target_col]].dropna()
    X = subset[feature_cols].copy()
    y = subset[target_col].copy()

    # Encode object / category columns
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



# ════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(results.keys())

    data = {label: [] for label in metric_labels}
    for name in model_names:
        for m, label in zip(metrics, metric_labels):
            val = results[name][m]
            data[label].append(round(val, 4) if val is not None else 0)

    fig = go.Figure()
    for i, label in enumerate(metric_labels):
        fig.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=data[label],
            marker_color=COLORS[i],
            text=[f"{v:.3f}" for v in data[label]],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        title="📊 Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_radar(results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        values = [res[m] for m in metrics]
        values += [values[0]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=name,
            line_color=COLORS[i % len(COLORS)],
            fillcolor=COLORS[i % len(COLORS)] + "26",
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template=PLOTLY_TEMPLATE,
        title="🕸️ Radar Chart – Model Metrics",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    return fig


def plot_roc_curves(results, y_test):
    n_classes = len(np.unique(y_test))
    if n_classes > 2:
        return None  # ROC curves for multi-class not shown per-class here
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash", color="#888"),
                  x0=0, x1=1, y0=0, y1=1)
    for i, (name, res) in enumerate(results.items()):
        if res["y_prob"] is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"][:, 1])
            auc_val = res["roc_auc"] or 0
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{name} (AUC={auc_val:.3f})",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
            ))
        except Exception:
            pass
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="📈 ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    return fig


def plot_confusion_matrices(results):
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    names = list(results.keys())

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.patch.set_facecolor("#1a1a2e")
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes[idx]
        cm = results[name]["cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            cmap=sns.color_palette("mako", as_cmap=True),
            linewidths=0.5,
            annot_kws={"size": 12, "color": "white"},
        )
        ax.set_title(name, color="white", fontsize=13, pad=10)
        ax.set_xlabel("Predicted", color="#a0a0c0")
        ax.set_ylabel("Actual", color="#a0a0c0")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a2e")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(pad=3)
    return fig


def plot_feature_importance(results, feature_names):
    tree_models = {
        name: res["model"]
        for name, res in results.items()
        if hasattr(res["model"], "feature_importances_")
    }
    if not tree_models:
        return None

    first_name, first_model = next(iter(tree_models.items()))
    importance = first_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(20)

    fig = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["#667eea", "#f5576c"],
        title=f"🌲 Feature Importance – {first_name}",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(fi_df) * 24),
    )
    return fig


def plot_data_distribution(df, target_col):
    value_counts = df[target_col].value_counts().reset_index()
    value_counts.columns = [target_col, "Count"]
    fig = px.pie(
        value_counts, names=target_col, values="Count",
        title=f"🎯 Target Distribution – '{target_col}'",
        color_discrete_sequence=COLORS,
        template=PLOTLY_TEMPLATE,
        hole=0.4,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400)
    return fig


def plot_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        title="🔥 Correlation Heatmap",
        template=PLOTLY_TEMPLATE,
        aspect="auto",
        zmin=-1, zmax=1,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=500)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Main app
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:30px 0 10px;">
        <h1 style="font-size:3rem; font-weight:800;
                   background:linear-gradient(90deg,#667eea,#f5576c,#43e97b);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🤖 ML Model Comparison Dashboard
        </h1>
        <p style="color:#a0a0c0; font-size:1.1rem;">
            Upload your dataset · Select features · Train &amp; compare classic ML models
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "📂 Upload Dataset",
            type=["csv", "xlsx", "xls"],
            help="CSV or Excel files are supported",
        )

        if uploaded_file is None:
            st.info("👆 Upload a file to get started!")
            st.markdown("---")
            st.markdown("### 💡 Quick Guide")
            st.markdown("""
            1. Upload a CSV / Excel file  
            2. Select feature columns  
            3. Pick the target column  
            4. Configure the train/test split  
            5. Choose models to compare  
            6. Hit **Train Models** 🚀
            """)
            return

        df = load_data(uploaded_file)
        if df is None:
            return

        st.success(f"✅ Loaded **{df.shape[0]}** rows × **{df.shape[1]}** columns")
        st.markdown("---")

        all_cols = df.columns.tolist()
        target_col = st.selectbox("🎯 Target Column", all_cols, index=len(all_cols) - 1)

        default_features = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect(
            "📋 Feature Columns",
            [c for c in all_cols if c != target_col],
            default=default_features,
        )

        st.markdown("---")
        test_size = st.slider("🔀 Test Set Size", 0.1, 0.5, 0.2, 0.05,
                              help="Fraction of data used for testing")
        st.markdown(f"Train: **{int((1-test_size)*100)}%**  |  Test: **{int(test_size*100)}%**")
        st.markdown("---")

        all_models = list(get_models().keys())
        selected_model_names = st.multiselect(
            "🧠 Models to Train",
            all_models,
            default=all_models,
        )
        st.markdown("---")

        train_btn = st.button("🚀 Train Models", use_container_width=True)

    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column.")
        return

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_explore, tab_results, tab_viz, tab_export = st.tabs([
        "🔍 Data Exploration",
        "📊 Model Results",
        "📈 Visualizations",
        "📥 Export",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 – Data Exploration
    # ════════════════════════════════════════════════════════════════════════
    with tab_explore:
        st.markdown('<p class="section-header">Data Exploration</p>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Rows",    f"{df.shape[0]:,}")
        c2.metric("📐 Columns", f"{df.shape[1]:,}")
        c3.metric("❌ Missing", f"{df.isnull().sum().sum():,}")
        c4.metric("📦 Duplicates", f"{df.duplicated().sum():,}")

        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### 📊 Descriptive Statistics")
            st.dataframe(df.describe(include="all").T, use_container_width=True)
        with col_r:
            st.markdown("### 🔍 Data Types & Missing Values")
            info_df = pd.DataFrame({
                "Type": df.dtypes.astype(str),
                "Non-Null": df.notnull().sum(),
                "Missing": df.isnull().sum(),
                "Missing %": (df.isnull().mean() * 100).round(2),
                "Unique": df.nunique(),
            })
            st.dataframe(info_df, use_container_width=True)

        st.markdown("### 🎯 Target Distribution")
        st.plotly_chart(plot_data_distribution(df, target_col),
                        use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            st.markdown("### 🔥 Correlation Heatmap")
            st.plotly_chart(plot_correlation_heatmap(df, numeric_cols),
                            use_container_width=True)

        if len(numeric_cols) >= 1:
            st.markdown("### 📦 Feature Distributions")
            sel_col = st.selectbox("Select column", numeric_cols, key="dist_col")
            fig_dist = px.histogram(
                df, x=sel_col, color_discrete_sequence=["#667eea"],
                marginal="box", template=PLOTLY_TEMPLATE,
                title=f"Distribution of {sel_col}",
            )
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # Tab 2 – Model Results  (populated after training)
    # ════════════════════════════════════════════════════════════════════════
    with tab_results:
        if not train_btn:
            st.info("👈 Configure your settings in the sidebar and click **Train Models** to begin.")
        else:
            if not selected_model_names:
                st.warning("⚠️ Please select at least one model.")
            else:
                with st.spinner("Preparing data…"):
                    try:
                        X, y = preprocess_data(df, feature_cols, target_col)
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42, stratify=y
                            )
                        except ValueError:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42
                            )
                    except Exception as e:
                        st.error(f"Data preparation failed: {e}")
                        return

                chosen_models = {n: m for n, m in get_models().items()
                                 if n in selected_model_names}

                st.markdown('<p class="section-header">Model Results</p>',
                            unsafe_allow_html=True)
                progress = st.progress(0)
                results = {}
                n_models = len(chosen_models)

                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc  = scaler.transform(X_test)
                n_classes  = len(np.unique(y_train))
                multiclass = n_classes > 2
                avg        = "weighted" if multiclass else "binary"

                for idx, (name, model) in enumerate(chosen_models.items()):
                    with st.spinner(f"Training {name}…"):
                        model.fit(X_train_sc, y_train)
                        y_pred = model.predict(X_test_sc)
                        y_prob = model.predict_proba(X_test_sc) \
                            if hasattr(model, "predict_proba") else None

                        roc_auc = None
                        if y_prob is not None:
                            try:
                                if multiclass:
                                    roc_auc = roc_auc_score(
                                        y_test, y_prob,
                                        multi_class="ovr", average="macro"
                                    )
                                else:
                                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                            except Exception:
                                roc_auc = None

                        results[name] = {
                            "model":     model,
                            "y_pred":    y_pred,
                            "y_prob":    y_prob,
                            "accuracy":  accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
                            "recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
                            "f1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
                            "roc_auc":   roc_auc,
                            "cm":        confusion_matrix(y_test, y_pred),
                        }
                    progress.progress((idx + 1) / n_models)

                st.success("✅ All models trained successfully!")
                st.session_state["results"]      = results
                st.session_state["y_test"]       = y_test
                st.session_state["feature_cols"] = feature_cols

                # ── Leaderboard ──────────────────────────────────────────────
                st.markdown("### 🏆 Leaderboard")
                metrics_df = pd.DataFrame([
                    {
                        "Model":       name,
                        "Accuracy":    round(res["accuracy"], 4),
                        "Precision":   round(res["precision"], 4),
                        "Recall":      round(res["recall"], 4),
                        "F1-Score":    round(res["f1"], 4),
                        "ROC-AUC":     round(res["roc_auc"], 4) if res["roc_auc"] is not None else None,
                    }
                    for name, res in results.items()
                ])
                metrics_df = metrics_df.sort_values("F1-Score", ascending=False)
                st.dataframe(metrics_df, use_container_width=True)

                # ── Best model highlight ──────────────────────────────────────
                best_name = max(results, key=lambda n: results[n]["f1"])
                best = results[best_name]
                st.markdown(f"### 🥇 Best Model: **{best_name}**")
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("Accuracy",  f"{best['accuracy']:.4f}")
                b2.metric("Precision", f"{best['precision']:.4f}")
                b3.metric("Recall",    f"{best['recall']:.4f}")
                b4.metric("F1-Score",  f"{best['f1']:.4f}")
                b5.metric("ROC-AUC",   f"{best['roc_auc']:.4f}" if best["roc_auc"] else "N/A")

                # ── Per-model details ─────────────────────────────────────────
                st.markdown("### 🔬 Detailed Classification Reports")
                for name, res in results.items():
                    with st.expander(f"📋 {name}"):
                        report = classification_report(y_test, res["y_pred"], output_dict=False)
                        st.code(report, language="text")

    # ════════════════════════════════════════════════════════════════════════
    # Tab 3 – Visualizations
    # ════════════════════════════════════════════════════════════════════════
    with tab_viz:
        results_ready = "results" in st.session_state
        if not results_ready:
            st.info("👈 Train models first to see visualizations.")
        else:
            results      = st.session_state["results"]
            y_test       = st.session_state["y_test"]
            feature_cols = st.session_state["feature_cols"]

            st.markdown('<p class="section-header">Visualizations</p>',
                        unsafe_allow_html=True)

            # Bar comparison
            st.plotly_chart(plot_metric_comparison(results), use_container_width=True)

            # Radar
            st.plotly_chart(plot_radar(results), use_container_width=True)

            # ROC curves
            roc_fig = plot_roc_curves(results, y_test)
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("ℹ️ ROC curves are shown for binary classification only.")

            # Confusion matrices
            st.markdown("### 🟦 Confusion Matrices")
            cm_fig = plot_confusion_matrices(results)
            st.pyplot(cm_fig, use_container_width=True)

            # Feature importance
            fi_fig = plot_feature_importance(results, feature_cols)
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.info("ℹ️ Feature importance is available for tree-based models (Random Forest, Gradient Boosting, Decision Tree).")

    # ════════════════════════════════════════════════════════════════════════
    # Tab 4 – Export
    # ════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown('<p class="section-header">Export Results</p>', unsafe_allow_html=True)

        if "results" not in st.session_state:
            st.info("👈 Train models first to export results.")
        else:
            results      = st.session_state["results"]
            y_test       = st.session_state["y_test"]
            feature_cols = st.session_state["feature_cols"]

            # ── CSV download ─────────────────────────────────────────────────
            metrics_rows = []
            for name, res in results.items():
                metrics_rows.append({
                    "Model":     name,
                    "Accuracy":  round(res["accuracy"], 6),
                    "Precision": round(res["precision"], 6),
                    "Recall":    round(res["recall"], 6),
                    "F1-Score":  round(res["f1"], 6),
                    "ROC-AUC":   round(res["roc_auc"], 6) if res["roc_auc"] is not None else None,
                })
            metrics_df = pd.DataFrame(metrics_rows)

            csv_buf = io.StringIO()
            metrics_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download Metrics CSV",
                data=csv_buf.getvalue(),
                file_name="ml_model_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # ── Text report download ──────────────────────────────────────────
            report_lines = ["ML Model Comparison Report\n", "=" * 60 + "\n\n"]
            for name, res in results.items():
                report_lines.append(f"Model: {name}\n")
                report_lines.append("-" * 40 + "\n")
                report_lines.append(f"  Accuracy : {res['accuracy']:.4f}\n")
                report_lines.append(f"  Precision: {res['precision']:.4f}\n")
                report_lines.append(f"  Recall   : {res['recall']:.4f}\n")
                report_lines.append(f"  F1-Score : {res['f1']:.4f}\n")
                if res["roc_auc"] is not None:
                    report_lines.append(f"  ROC-AUC  : {res['roc_auc']:.4f}\n")
                report_lines.append(
                    f"\nClassification Report:\n"
                    f"{classification_report(y_test, res['y_pred'])}\n\n"
                )
            report_text = "".join(report_lines)
            st.download_button(
                label="⬇️ Download Full Report (TXT)",
                data=report_text,
                file_name="ml_model_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown("### 📊 Metrics Preview")
            st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    main()
