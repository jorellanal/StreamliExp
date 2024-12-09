import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

tf.config.set_visible_devices([], 'GPU')
# Cargar el modelo preentrenado
model = tf.keras.models.load_model('Models/model.keras')

# Función para preprocesar el archivo cargado
def preprocess_data(uploaded_file):
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file)
    
    # Verificar si las columnas coinciden con las del modelo (por ejemplo, el modelo espera las mismas características)
    # Si es necesario, ajusta el DataFrame para coincidir con los datos de entrada de tu modelo
    X = df.drop(columns=['NOCOBRO'])  # Eliminar la columna de la etiqueta si está presente
    X_scaled = StandardScaler().fit_transform(X)  # Escalar las características
    
    return X_scaled

# Título de la aplicación
st.title("Clasificación con Modelo Preentrenado")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Preprocesar los datos del archivo subido
    X_scaled = preprocess_data(uploaded_file)
    
    # Realizar la predicción con el modelo
    predictions = model.predict(X_scaled)
    
    # Mostrar las predicciones
    st.write("Resultados de la predicción:")
    st.write(predictions)
    
    # Si deseas mostrar la clase predecida (0 o 1):
    predicted_classes = (predictions > 0.5).astype("int32")  # Umbral de 0.5 para clasificación binaria
    st.write("Clases predichas:")
    st.write(predicted_classes)

