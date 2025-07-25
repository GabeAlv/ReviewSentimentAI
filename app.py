import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from limpeza_dataset import limpar_texto

nltk.download('stopwords') # caso execute só este script

# carregando o modelo e o vetorizador
try:
    modelo = joblib.load('modelo.pkl')
    vetorizador = joblib.load('vetorizador.pkl')
except FileNotFoundError:
    print('Arquivos não encotrados')

#interface
st.title('Analisador de sentimentos de review de produtos')
st.write('Digite uma review de um produto em português e o modelo vai prever o sentimento (positivo ou negativo)')

#entrada do usuario
texto_input = st.text_area('texto para analise: ')

#removendo stopwords do português
if texto_input:
    #pré processamento
    texto_limpo = limpar_texto(texto_input)
    stop_words = set(stopwords.words('portuguese'))
    texto_limpo = ' '.join([word for word in texto_limpo.split() if word not in stop_words])

    #vetorização
    texto_vetorizado = vetorizador.transform([texto_limpo])

    #predição
    sentimento = modelo.predict(texto_vetorizado)[0]

    st.subheader('O resultado é: ')
    st.success(f'O sentimento desta review é **{sentimento.upper()}** ✅ (Ótima)' if sentimento == 'Positivo' else f'O sentimento desta review é **{sentimento.upper()}**❌ (Ruim)')