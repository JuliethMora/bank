import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from joblib import load as joblib_load
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ==============================
# Configuración Streamlit
# ==============================
st.set_page_config(page_title="Bank Marketing Prediction", page_icon="📊", layout="wide")
st.title("📊 Bank Marketing Predictive System")

# ==============================
# Ignorar warnings de versión
# ==============================
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ==============================
# Función para cargar modelos de forma segura
# ==============================
def load_model(model_path: str):
    """
    Carga un modelo usando joblib.
    """
    if not os.path.exists(model_path):
        st.error(f"❌ Archivo de modelo no encontrado: {model_path}")
        st.stop()
    try:
        return joblib_load(model_path)
    except Exception as e:
        st.error(f"❌ Error cargando el modelo {model_path}: {e}")
        st.stop()

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

# Escalador global para los modelos
scaler = MinMaxScaler()
scaler.fit(data.drop("y", axis=1))

# ==============================
# Preprocesamiento
# ==============================
def preprocess(df):
    df_clean = df.copy()
    le = LabelEncoder()
    for col in df_clean.select_dtypes(include=["object"]).columns:
        try:
            df_clean[col] = le.fit_transform(df_clean[col])
        except:
            pass
    X_scaled = scaler.transform(df_clean.drop("y", axis=1))
    return X_scaled, df_clean["y"]

# ==============================
# Función genérica para evaluar modelo
# ==============================
def evaluar_modelo(model_path: str, df: pd.DataFrame):
    """
    Carga el modelo, predice sobre df y devuelve métricas y predicciones.
    """
    model = load_model(model_path)
    X, y = preprocess(df)
    
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds
    cm = confusion_matrix(y, preds)
    fpr, tpr, _ = roc_curve(y, proba) if hasattr(model, "predict_proba") else ([], [], [])
    report_dict = classification_report(y, preds, output_dict=True)
    
    feature_importances = model.feature_importances_ if hasattr(model, "feature_importances_") else None
    
    return {
        "model": model,
        "preds": preds,
        "proba": proba,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "accuracy": accuracy_score(y, preds),
        "auc": roc_auc_score(y, proba) if hasattr(model, "predict_proba") else None,
        "report": report_dict,
        "feature_importances": feature_importances,
        "y_true": y
    }

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

# ==============================
# Evaluación automática con dataset base
# ==============================
st.header("📈 Evaluación Automática con Dataset Base")
metrics_all = {}
for name, path in models.items():
    metrics_all[name] = evaluar_modelo(path, data)

# Métricas principales usando st.metric
st.subheader(f"Métricas principales: {selected_model_name}")
selected_metrics = metrics_all[selected_model_name]
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{selected_metrics['accuracy']*100:.2f}%")
col2.metric("AUC", f"{selected_metrics['auc']*100:.2f}%" if selected_metrics['auc'] else "N/A")
col3.metric("Total Registros", f"{len(data):,}")

# ==============================
# ROC comparada
# ==============================
st.subheader("ROC Comparada")
fig = go.Figure()
for name, metrics in metrics_all.items():
    if metrics['fpr'] != []:
        fig.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines',
                                 name=f"{name} (AUC={metrics['auc']:.3f})"))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
fig.update_layout(title="ROC Curve Comparada", xaxis_title="False Positive Rate",
                  yaxis_title="True Positive Rate", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ==============================
# Confusion Matrix
# ==============================
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

# ==============================
# Feature Importance
# ==============================
st.subheader("🔍 Feature Importance")
if selected_metrics['feature_importances'] is not None:
    importances = selected_metrics['feature_importances']
    features = data.drop("y", axis=1).columns
    fi_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title="Feature Importance",
                 color="Importance", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# Estadísticas descriptivas
# ==============================
st.header("📊 Estadísticas descriptivas")
st.dataframe(data.describe())

# Histogramas interactivos
st.subheader("Distribución de Variables")
for col in data.select_dtypes(include=np.number).columns:
    fig = px.histogram(data, x=col, nbins=50, title=f"Distribución de {col}")
    st.plotly_chart(fig, use_container_width=True)

# Distribución de y
fig = px.histogram(data, x="y", color="y", title="Distribución de la variable objetivo 'y'")
st.plotly_chart(fig, use_container_width=True)
