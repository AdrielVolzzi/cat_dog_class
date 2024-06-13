import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Carregar o modelo treinado no formato recomendado
model = load_model('cat_dog_classifier.keras')

# Função para fazer a previsão
def predict(image):
    # Converter a imagem para tons de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Redimensionar a imagem
    resized_image = cv2.resize(gray_image, (150, 150))
    # Normalizar a imagem
    normalized_image = resized_image / 255.0
    # Expandir as dimensões para incluir o batch size
    input_image = np.expand_dims(normalized_image, axis=(0, -1))
    # Fazer a previsão
    prediction = model.predict(input_image)
    return prediction

# Configurar o título do app
st.title('Cachorro ou Gato?')

# Configurar o cabeçalho do app
st.header('Faça o upload das imagens para classifica-las como Cachorro ou Gato')

# Carregar as imagens do usuário
uploaded_files = st.file_uploader("Selecione as imagens...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Ler a imagem
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Converter a imagem para um formato utilizável
        image = np.array(image)
        
        # Prever a classe da imagem
        prediction = predict(image)
        label = 'Cachorro' if prediction > 0.5 else 'Gato'
        confidence = float(prediction) if label == 'Cachorro' else 1 - float(prediction)
        
        # Exibir a previsão e a confiança
        st.write(f'A imagem foi classificada como: **{label}** com uma confiança de **{confidence*100:.2f}%**')
