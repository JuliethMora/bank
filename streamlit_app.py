import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from joblib import load as joblib_load

# ===================================
#       STREAMLIT CONFIGURATION
# ===================================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Bank Marketing Predictive System – Automatic Evaluation")

# ===================================
#       GENERIC MODEL LOADER
# ===================================
def load_model(model_path):
    if not os.path.exists(model_path):
        return None, f"❌ Not found: {model_path}"
    try:
        model = joblib_load(model_path)
        return model, None
    except Exception as e:
        return None, f"❌ Error loading {model_path}: {e}"

# ===================================
#          AVAILABLE MODELS
# ===================================
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Optimized Gradient Boosting": "optimized_gradient_boosting_model.pkl"
}

# ===================================
#          LOAD BASE DATASET
# ===================================
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional-full.csv", sep=";")
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])
    return df

data = load_data()
X_base = data.drop("y", axis=1)
y_base = data["y"]

# Global scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_base)

# ===================================
#      MAIN METRICS CARDS
# ===================================
st.subheader("📊 Dataset Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Unique Clients", f"{data['y'].shape[0]}")
col2.metric("Total Records", f"{data.shape[0]:,}")
col3.metric("Proportion y=1", f"{y_base.sum()/len(y_base):.2%}")

# ===================================
#      VARIABLE DASHBOARD
# ===================================
st.subheader("Variable Distributions")
for col in X_base.columns[:10]:  # limit columns to avoid overcrowding
    fig = px.histogram(data, x=col, nbins=30, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Target Variable 'y' Distribution")
fig = px.histogram(data, x="y", title="Target Variable Distribution")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Descriptive Statistics")
st.dataframe(data.describe())

# ===================================
#        MODEL EVALUATION
# ===================================
st.header("📊 Automatic Model Evaluation")
metrics_all = {}
failed_models = []

for name, path in model_files.items():
    model, error = load_model(path)
    if model is None:
        failed_models.append(error)
        continue

    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else preds

    acc = accuracy_score(y_base, preds)
    auc = roc_auc_score(y_base, proba) if len(np.unique(y_base)) > 1 else 0
    cm = confusion_matrix(y_base, preds)
    fpr, tpr, _ = roc_curve(y_base, proba) if len(np.unique(y_base)) > 1 else ([], [], [])

    metrics_all[name] = {
        "accuracy": acc,
        "auc": auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "feature_importances": getattr(model, "feature_importances_", None)
    }

# Show loading errors
if failed_models:
    st.warning("Some models could not be loaded:")
    for msg in failed_models:
        st.warning(msg)

# ===================================
#       MAIN METRICS USING st.metric
# ===================================
st.subheader("📈 Main Model Metrics")
cols = st.columns(len(metrics_all))
for col, (name, m) in zip(cols, metrics_all.items()):
    col.metric(label=f"{name} - Accuracy", value=f"{m['accuracy']:.4f}")
    col.metric(label=f"{name} - AUC", value=f"{m['auc']:.4f}")

# ===================================
#       ROC Curves Interactive with Plotly
# ===================================
st.subheader("📊 ROC Curves")
fig = go.Figure()
for name, m in metrics_all.items():
    if len(m['fpr']) > 0:
        fig.add_trace(go.Scatter(x=m['fpr'], y=m['tpr'], mode='lines', name=name))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig, use_container_width=True)

# ===================================
#      Accuracy Comparison
# ===================================
st.subheader("📊 Model Accuracy Comparison")
acc_df = pd.DataFrame({
    "Model": list(metrics_all.keys()),
    "Accuracy": [m["accuracy"] for m in metrics_all.values()]
})
fig = px.bar(acc_df, x="Model", y="Accuracy", text="Accuracy", title="Accuracy by Model")
st.plotly_chart(fig, use_container_width=True)

# ===================================
#      Feature Importance (if available)
# ===================================
st.subheader("🔍 Feature Importance (for models that support it)")
for name, m in metrics_all.items():
    if m['feature_importances'] is not None:
        fi_df = pd.DataFrame({
            "Feature": X_base.columns,
            "Importance": m['feature_importances']
        }).sort_values(by="Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title=f"Feature Importance - {name}")
        st.plotly_chart(fig, use_container_width=True)
