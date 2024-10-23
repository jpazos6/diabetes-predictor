import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carga de datos
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes.csv")
    return data

# Entrenamiento del modelo
@st.cache_resource
def train_model(data):
    X = data[["Pregnancies", "Glucose", "Insulin", "BMI", "Age"]]
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Interfaz de Streamlit
st.title("Predicción de Diabetes")

st.write("""
Esta aplicación predice si un paciente tiene diabetes basado en ciertas características médicas.
""")

data = load_data()
model, accuracy = train_model(data) 

st.write(f"Precisión del modelo: {accuracy:.2f}")

st.sidebar.header("Parámetros del paciente")

def user_input_features():
    pregnancies = st.sidebar.number_input("Embarazos", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucosa", min_value=0, max_value=200, value=120)
    insulin = st.sidebar.number_input("Insulina", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=70.0, value=25.0)
    age = st.sidebar.number_input("Edad", min_value=1, max_value=120, value=30)
    data = {"Pregnancies": pregnancies,
            "Glucose": glucose,
            "Insulin": insulin,
            "BMI": bmi,
            "Age": age}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("Parámetros del usuario")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Predicción")
st.write("Diabético" if prediction[0] == 1 else "No diabético")

st.subheader("Probabilidad de predicción")
st.write(prediction_proba)

