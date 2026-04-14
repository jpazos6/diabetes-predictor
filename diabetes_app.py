import streamlit as st
import pandas as pd
import joblib

# ── Carga del modelo y el scaler exportados desde el notebook ────────────────
# El preprocesamiento aplicado aquí es idéntico al del entrenamiento:
# mismas columnas, mismo MinMaxScaler ajustado sobre los datos de train.
@st.cache_resource
def load_artifacts():
    model   = joblib.load("model.pkl")
    scaler  = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ── Interfaz ─────────────────────────────────────────────────────────────────
st.title("Predicción de Diabetes")
st.write("""
Esta aplicación predice si un paciente tiene diabetes basándose en
características médicas. Utiliza un modelo Random Forest entrenado sobre
el dataset Pima Indians Diabetes (solo mujeres ≥ 21 años de ascendencia india).
""")

st.sidebar.header("Parámetros del paciente")

def user_input_features():
    pregnancies = st.sidebar.number_input("Embarazos",                      min_value=0,   max_value=20,  value=1)
    glucose     = st.sidebar.number_input("Glucosa (mg/dL)",                min_value=0,   max_value=200, value=120)
    insulin     = st.sidebar.number_input("Insulina (µIU/mL)",              min_value=0,   max_value=900, value=80)
    bmi         = st.sidebar.number_input("Índice de Masa Corporal (BMI)",  min_value=0.0, max_value=70.0,value=25.0)
    age         = st.sidebar.number_input("Edad",                           min_value=21,  max_value=120, value=30)

    return pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose":     [glucose],
        "Insulin":     [insulin],
        "BMI":         [bmi],
        "Age":         [age],
    })

input_df = user_input_features()

st.subheader("Parámetros introducidos")
st.write(input_df)

# ── Preprocesamiento: aplicar el mismo scaler del entrenamiento ───────────────
input_scaled = scaler.transform(input_df)

# ── Predicción ───────────────────────────────────────────────────────────────
prediction       = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Resultado")
if prediction[0] == 1:
    st.error("Diabético")
else:
    st.success("No diabético")

st.subheader("Probabilidad de predicción")
proba_df = pd.DataFrame(prediction_proba, columns=["No diabético", "Diabético"])
st.write(proba_df)
