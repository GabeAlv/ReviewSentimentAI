import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
import re


nltk.download('stopwords') # caso execute só este script

modelo = None
vetorizador = None

# carregando o modelo e o vetorizador
try:
    modelo = joblib.load('Analisador-sentimento/modelo.pkl')
    vetorizador = joblib.load('Analisador-sentimento/vetorizador.pkl')
except FileNotFoundError:
    print('Arquivos não encotrados')

#interface do streamlit
st.title('Analisador de sentimentos de review de produtos')
st.write('Digite uma review de um produto em português e o modelo vai prever o sentimento (positivo ou negativo)')

#entrada do usuario
texto_input = st.text_area('texto para analise: ')

def limpar_texto(texto):
  texto = re.sub(r'@\w+', '', texto) # remove menções
  texto = re.sub(r'https?://\S+', '', texto) # remove URLs
  texto = re.sub(r'[^a-zA-ZÀ-ÿ]', ' ', texto) # remove caracteres nao alfabéticos
  texto = texto.lower() # colocando os textos no minusculo
  return texto

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