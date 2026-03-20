import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.models import train_models
from src.evaluate import display_metrics
from src.plots import plot_results
from src.reporting import generate_report

def main():
    st.title("ML Model Comparison Dashboard")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write(data)
        
        target = st.selectbox("Select target variable", data.columns)
        
        # Auto-detect problem type, preprocess, train, and evaluate the models
        problem_type = "Classification" if data[target].dtype == 'object' else "Regression"
        
        X, y = preprocess_data(data, target)
        models = train_models(X, y, problem_type)
        
        report = generate_report(models, y)
        st.write(report)
        
        plot_results(models)

if __name__ == "__main__":
    main()