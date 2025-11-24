import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from joblib import load as joblib_load
import plotly.express as px
import plotly.graph_objects as go
from sklearn.exceptions import InconsistentVersionWarning

# ==============================
# Configuración Streamlit
# ==============================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Bank Marketing Predictive System")

# ==============================
# Ignorar warnings de versión
# ==============================
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ==============================
# Función para cargar modelos
# ==============================
def load_model(modelo_path: str):
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

# ==============================
# Modelos disponibles
# ==============================
models = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Optimized Gradient Boosting": "optimized_gradient_boosting_model.pkl"
}

st.sidebar.header("🛠️ Modelos Disponibles")
selected_model_name = st.sidebar.selectbox("Selecciona un modelo:", list(models.keys()))
selected_model_path = models[selected_model_name]
model = load_model(selected_model_path)

# ==============================
# Cargar dataset base
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional-full.csv", sep=";")
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])
    return df

data = load_data()
scaler = MinMaxScaler()
scaler.fit(data.drop("y", axis=1))

# ==============================
# Preprocesamiento
# ==============================
def preprocess(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = le.fit_transform(df[col])
        except:
            pass
    return scaler.transform(df)

# ==============================
# Métricas y gráficos
# ==============================
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

# ==============================
# Dataset base: estadísticas descriptivas
# ==============================
st.header("📊 Estadísticas Descriptivas del Dataset Base")

# Histogramas interactivos
for col in data.select_dtypes(include=np.number).columns:
    fig = px.histogram(data, x=col, nbins=50, title=f"Distribución de {col}")
    st.plotly_chart(fig, use_container_width=True)

# Distribución de y
fig = px.histogram(data, x="y", color="y", title="Distribución de la variable objetivo 'y'")
st.plotly_chart(fig, use_container_width=True)

# Tabla descriptiva
st.dataframe(data.describe())

# ==============================
# Evaluación automática con dataset base
# ==============================
st.header("📈 Evaluación Automática de Modelos con Dataset Base")

X_base = preprocess(data.drop("y", axis=1))
y_base = data["y"]

metrics_all = {}
for name, path in models.items():
    this_model = load_model(path)
    metrics_all[name] = compute_metrics(this_model, X_base, y_base)

# Mostrar métricas principales usando st.metric
st.subheader("Métricas Principales")
col1, col2, col3 = st.columns(3)
selected_metrics = metrics_all[selected_model_name]
col1.metric("Accuracy", f"{selected_metrics['accuracy']*100:.2f}%")
col2.metric("AUC", f"{selected_metrics['auc']*100:.2f}%")
col3.metric("Total Registros", f"{len(data):,}")

# ROC comparada con Plotly
fig = go.Figure()
for name, metrics in metrics_all.items():
    fig.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines',
                             name=f"{name} (AUC={metrics['auc']:.3f})"))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
fig.update_layout(title="ROC Curve Comparada", xaxis_title="False Positive Rate",
                  yaxis_title="True Positive Rate", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = selected_metrics['cm']
cm_fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=["Predicted 0", "Predicted 1"],
    y=["Actual 0", "Actual 1"],
    colorscale="Blues",
    text=cm,
    texttemplate="%{text}"
))
cm_fig.update_layout(title="Confusion Matrix", template="plotly_white")
st.plotly_chart(cm_fig, use_container_width=True)

# Feature Importance
st.subheader("🔍 Feature Importance")
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    features = data.drop("y", axis=1).columns
    fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title="Feature Importance", color="Importance",
                 color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Subida opcional de CSV
# ==============================
st.sidebar.header("📥 Cargar dataset para evaluar")
uploaded_file = st.sidebar.file_uploader("Selecciona un CSV", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file, sep=";")
    st.subheader("📄 Datos cargados")
    st.write(test_df.head())

    if "y" not in test_df.columns:
        st.warning("⚠️ El dataset no contiene columna 'y'. Solo se harán predicciones.")
        X_test = preprocess(test_df)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        st.dataframe(pd.DataFrame({"Prediction": preds, "Probability": proba}))
    else:
        y_test = test_df["y"].map({"yes":1, "no":0}) if test_df["y"].dtype=="object" else test_df["y"]
        X_test = preprocess(test_df.drop("y", axis=1))

        st.header("📊 Evaluación de Modelos con Dataset Subido")
        metrics_all_uploaded = {}
        for name, path in models.items():
            this_model = load_model(path)
            metrics_all_uploaded[name] = compute_metrics(this_model, X_test, y_test)

        # Métricas principales tipo st.metric
        st.subheader("Métricas Principales (CSV Subido)")
        selected_metrics_uploaded = metrics_all_uploaded[selected_model_name]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{selected_metrics_uploaded['accuracy']*100:.2f}%")
        col2.metric("AUC", f"{selected_metrics_uploaded['auc']*100:.2f}%")
        col3.metric("Total Registros", f"{len(test_df):,}")

        # ROC comparada con Plotly
        fig = go.Figure()
        for name, metrics in metrics_all_uploaded.items():
            fig.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines',
                                     name=f"{name} (AUC={metrics['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
        fig.update_layout(title="ROC Curve Comparada (CSV Subido)", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
