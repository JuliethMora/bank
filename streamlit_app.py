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
#       CONFIGURACIÓN STREAMLIT
# ===================================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Bank Marketing Predictive System – Evaluación Automática")

# ===================================
#       FUNCIÓN GENÉRICA PARA CARGAR MODELOS
# ===================================
def load_model(model_path):
    if not os.path.exists(model_path):
        return None, f"❌ No encontrado: {model_path}"
    try:
        model = joblib_load(model_path)
        return model, None
    except Exception as e:
        return None, f"❌ Error cargando {model_path}: {e}"

# ===================================
#          MODELOS DISPONIBLES
# ===================================
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Optimized Gradient Boosting": "optimized_gradient_boosting_model.pkl"
}

# ===================================
#          CARGAR DATASET BASE
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

# Escalador global
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_base)

# ===================================
#      TARJETAS PRINCIPALES
# ===================================
st.subheader("📊 Resumen General del Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Clientes Únicos", f"{data['y'].shape[0]}")
col2.metric("Total Registros", f"{data.shape[0]:,}")
col3.metric("Proporción y=1", f"{y_base.sum()/len(y_base):.2%}")

# ===================================
#      DASHBOARD DE VARIABLES
# ===================================
st.subheader("Distribución de Variables")
for col in X_base.columns[:10]:  # limitar algunas columnas para no saturar
    fig = px.histogram(data, x=col, nbins=30, title=f"Distribución de {col}")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Distribución de la Variable Objetivo 'y'")
fig = px.histogram(data, x="y", title="Distribución de y")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Estadísticas Descriptivas")
st.dataframe(data.describe())

# ===================================
#        EVALUACIÓN DE MODELOS
# ===================================
st.header("📊 Evaluación Automática de Modelos")
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

# Mostrar errores de carga
if failed_models:
    st.warning("Algunos modelos no se pudieron cargar:")
    for msg in failed_models:
        st.warning(msg)

# ===================================
#       Métricas principales tipo st.metric
# ===================================
st.subheader("📈 Métricas Principales de Modelos")
cols = st.columns(len(metrics_all))
for col, (name, m) in zip(cols, metrics_all.items()):
    col.metric(label=f"{name} - Accuracy", value=f"{m['accuracy']:.4f}")
    col.metric(label=f"{name} - AUC", value=f"{m['auc']:.4f}")

# ===================================
#       ROC Curves interactivo con Plotly
# ===================================
st.subheader("📊 Curvas ROC de los Modelos")
fig = go.Figure()
for name, m in metrics_all.items():
    if len(m['fpr']) > 0:
        fig.add_trace(go.Scatter(x=m['fpr'], y=m['tpr'], mode='lines', name=name))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig, use_container_width=True)

# ===================================
#      Accuracy Comparación
# ===================================
st.subheader("📊 Comparación de Accuracy entre Modelos")
acc_df = pd.DataFrame({
    "Modelo": list(metrics_all.keys()),
    "Accuracy": [m["accuracy"] for m in metrics_all.values()]
})
fig = px.bar(acc_df, x="Modelo", y="Accuracy", text="Accuracy", title="Accuracy por Modelo")
st.plotly_chart(fig, use_container_width=True)

# ===================================
#      Feature Importance (si existe)
# ===================================
st.subheader("🔍 Feature Importance (para modelos que lo soportan)")
for name, m in metrics_all.items():
    if m['feature_importances'] is not None:
        fi_df = pd.DataFrame({
            "Feature": X_base.columns,
            "Importance": m['feature_importances']
        }).sort_values(by="Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title=f"Feature Importance - {name}")
        st.plotly_chart(fig, use_container_width=True)
