import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# ===================================
#       CONFIGURACIÓN STREAMLIT
# ===================================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Bank Marketing Predictive System – Dashboard Completo")

# ===================================
#          CARGA DE MODELOS
# ===================================
def load_pickle_model(file):
    return pickle.load(open(file, "rb"))

# Modelos iniciales
models = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Gradient Boosting (Initial)": "gradient_boosting_model.pkl",
    "Gradient Boosting (Optimized)": "optimized_gradient_boosting_model.pkl"
}

# ===================================
#    PIPELINE PARA SUBIR NUEVOS MODELOS
# ===================================
st.sidebar.header("🛠️ Modelos Disponibles")

uploaded_model = st.sidebar.file_uploader("📥 Subir modelo .pkl", type=["pkl"])

if uploaded_model:
    model_name = uploaded_model.name.replace(".pkl", "")
    save_path = os.path.join("uploaded_models", uploaded_model.name)
    os.makedirs("uploaded_models", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    models[model_name] = save_path
    st.sidebar.success(f"Modelo '{model_name}' cargado y registrado.")

selected_model_name = st.sidebar.selectbox("Selecciona un modelo:", list(models.keys()))
selected_model_path = models[selected_model_name]
model = load_pickle_model(selected_model_path)

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

# Scaler global
scaler = MinMaxScaler()
scaler.fit(data.drop("y", axis=1))

# ===================================
#       DASHBOARD VISUAL COMPLETO
# ===================================
st.header("📊 Exploratory Data Analysis Dashboard")

tab1, tab2, tab3 = st.tabs(["Distribuciones", "Correlaciones", "Estadísticas"])

with tab1:
    st.subheader("Histogramas")
    fig, ax = plt.subplots(figsize=(10, 5))
    data.hist(ax=ax)
    st.pyplot(fig)

    st.subheader("Distribución de Y")
    fig = plt.figure(figsize=(5, 4))
    sns.countplot(data["y"])
    st.pyplot(fig)

with tab2:
    st.subheader("Heatmap de correlaciones")
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=False, cmap="coolwarm")
    st.pyplot(fig)

with tab3:
    st.subheader("Estadísticas descriptivas")
    st.write(data.describe())

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
        "report": classification_report(y_true, preds)
    }

def plot_cm(cm):
    fig = plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

def plot_roc_curve(models_dict):
    fig = plt.figure(figsize=(6, 5))
    for name, metrics in models_dict.items():
        plt.plot(metrics["fpr"], metrics["tpr"], label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.legend()
    plt.title("ROC Comparison")
    st.pyplot(fig)

# ===================================
#     SI EL USUARIO SUBE UN DATASET
# ===================================
if uploaded_file:
    test_df = pd.read_csv(uploaded_file, sep=";")
    st.subheader("📄 Datos cargados")
    st.write(test_df.head())

    if "y" not in test_df.columns:
        st.warning("⚠️ El dataset no contiene columna 'y'. Solo se harán predicciones.")
        X_test = preprocess(test_df)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        st.write(pd.DataFrame({"Prediction": preds, "Probability": proba}))
    else:
        # Preprocess
        y_test = test_df["y"].map({"yes":1, "no":0}) if test_df["y"].dtype=="object" else test_df["y"]
        X_test = preprocess(test_df.drop("y", axis=1))

        # ================================
        #   COMPARACIÓN ENTRE MODELOS
        # ================================
        st.header("📊 Comparación entre modelos")
        metrics_all = {}

        for name, path in models.items():
            this_model = load_pickle_model(path)
            metrics_all[name] = compute_metrics(this_model, X_test, y_test)

        # Tabla comparativa
        comp_table = pd.DataFrame({
            name: {
                "Accuracy": m["accuracy"],
                "AUC": m["auc"]
            }
            for name, m in metrics_all.items()
        }).T

        st.write(comp_table)

        # Gráfica de barras
        fig = plt.figure(figsize=(6, 4))
        sns.barplot(x=comp_table.index, y=comp_table["Accuracy"])
        plt.title("Accuracy Comparison")
        plt.xticks(rotation=30)
        st.pyplot(fig)

        # ROC comparado
        plot_roc_curve(metrics_all)

        # Métricas del modelo seleccionado
        st.subheader(f"📌 Evaluación detallada: {selected_model_name}")
        selected_metrics = metrics_all[selected_model_name]

        st.code(selected_metrics["report"])

        st.subheader("Confusion Matrix")
        plot_cm(selected_metrics["cm"])

        # Feature Importance
        st.subheader("🔍 Feature Importance")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            features = test_df.drop("y", axis=1).columns

            fig = plt.figure(figsize=(7, 5))
            sns.barplot(x=importances, y=features)
            plt.title("Feature Importance")
            st.pyplot(fig)

else:
    st.info("Sube un archivo CSV para comenzar el análisis.")
