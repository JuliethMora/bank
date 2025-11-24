import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from joblib import load as joblib_load

# ===================================
#       CONFIGURACIÓN STREAMLIT
# ===================================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Bank Marketing Predictive System")

# ===================================
#       FUNCIÓN PARA CARGAR MODELOS
# ===================================
def load_model(modelo_path: str):
    """Carga un modelo desde disco usando joblib o pickle de forma segura."""
    if not os.path.exists(modelo_path):
        st.error(f"❌ Archivo de modelo no encontrado: {modelo_path}")
        st.stop()
    try:
        return joblib_load(modelo_path)
    except Exception:
        import pickle
        try:
            with open(modelo_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"❌ Error cargando el modelo: {e}")
            st.stop()

# ===================================
#       CARGA DE MODELOS
# ===================================
models = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Optimized Gradient Boosting": "optimized_gradient_boosting_model.pkl"
}

st.sidebar.header("🛠️ Modelos Disponibles")
selected_model_name = st.sidebar.selectbox("Selecciona un modelo:", list(models.keys()))
selected_model_path = models[selected_model_name]
model = load_model(selected_model_path)

# ===================================
#       CARGAR DATASET BASE
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

# Escalador global
scaler = MinMaxScaler()
scaler.fit(data.drop("y", axis=1))

# ===================================
#       ANÁLISIS DESCRIPTIVO SECUENCIAL
# ===================================
st.header("📊 Exploratory Data Analysis")

# --- Paso 1: Métricas rápidas ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Clientes", value=f"{len(data)}")
col2.metric("Total Variables", value=f"{data.shape[1]}")
col3.metric("Proporción Y=1", value=f"{data['y'].mean():.2%}")

# --- Paso 2: Histogramas ---
st.subheader("Distribuciones de las Variables")
for col in data.drop("y", axis=1).columns[:10]:  # Mostrar solo primeras 10 para no saturar
    fig = px.histogram(data, x=col, nbins=30, title=f"Distribución de {col}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- Paso 3: Distribución de Y ---
st.subheader("Distribución de la Variable Objetivo 'y'")
fig = px.histogram(data, x="y", color="y", title="Distribución de Y", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Paso 4: Correlaciones ---
st.subheader("Correlaciones entre Variables")
fig = px.imshow(data.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
st.plotly_chart(fig, use_container_width=True)

# --- Paso 5: Estadísticas descriptivas ---
st.subheader("Estadísticas Descriptivas")
st.dataframe(data.describe())

# ===================================
#     UPLOAD DATA PARA PREDICT / EVAL
# ===================================
st.sidebar.header("📥 Cargar dataset para evaluar")
uploaded_file = st.sidebar.file_uploader("Selecciona un CSV", type=["csv"])

# ===================================
#       PREPROCESSING DEL INPUT
# ===================================
def preprocess(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = le.fit_transform(df[col])
        except:
            pass
    return scaler.transform(df)

# ===================================
#      FUNCIÓN DE MÉTRICAS Y GRÁFICAS
# ===================================
def compute_metrics(model, X, y_true):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y_true, preds)
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "auc": roc_auc_score(y_true, proba),
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "preds": preds,
        "proba": proba,
        "report": classification_report(y_true, preds, output_dict=True)
    }

def plot_cm(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

def plot_roc_curve(models_dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, metrics in models_dict.items():
        ax.plot(metrics["fpr"], metrics["tpr"], label=f"{name} (AUC={metrics['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison")
    ax.legend()
    st.pyplot(fig)

# ===================================
#     SI EL USUARIO SUBE UN DATASET
# ===================================
if uploaded_file:
    test_df = pd.read_csv(uploaded_file, sep=";")
    st.subheader("📄 Datos cargados")
    st.dataframe(test_df.head())

    if "y" not in test_df.columns:
        st.warning("⚠️ El dataset no contiene columna 'y'. Solo se harán predicciones.")
        X_test = preprocess(test_df)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        st.subheader("Predicciones")
        st.dataframe(pd.DataFrame({"Prediction": preds, "Probability": proba}))
    else:
        y_test = test_df["y"].map({"yes":1, "no":0}) if test_df["y"].dtype=="object" else test_df["y"]
        X_test = preprocess(test_df.drop("y", axis=1))

        # Comparación de modelos
        st.header("📊 Comparación entre Modelos")
        metrics_all = {}
        for name, path in models.items():
            this_model = load_model(path)
            metrics_all[name] = compute_metrics(this_model, X_test, y_test)

        # Tabla comparativa con metricas visuales
        comp_table = pd.DataFrame({
            name: {"Accuracy": m["accuracy"], "AUC": m["auc"]}
            for name, m in metrics_all.items()
        }).T
        st.dataframe(comp_table.style.background_gradient(cmap="Blues"))

        # ROC Comparado
        st.subheader("ROC Comparado")
        plot_roc_curve(metrics_all)

        # Métricas del modelo seleccionado
        st.subheader(f"📌 Evaluación detallada: {selected_model_name}")
        selected_metrics = metrics_all[selected_model_name]
        st.dataframe(pd.DataFrame(selected_metrics["report"]).transpose())

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        plot_cm(selected_metrics["cm"])

        # Feature Importance
        st.subheader("🔍 Feature Importance")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            features = test_df.drop("y", axis=1).columns
            fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(fi_df["Feature"], fi_df["Importance"], color="skyblue")
            ax.set_title("Feature Importance")
            st.pyplot(fig)

else:
    st.info("Sube un archivo CSV para comenzar el análisis.")
