# ReviewSentimentAI
Classificador de sentimentos com Naive Bayes, que identifica se a review expressa opinião positiva ou negativa.

Este projeto utiliza técnicas de Processamento de Linguagem Natural (NLP) e aprendizado de máquina para classificar textos automaticamente com base em seus padrões linguísticos. 
A interface foi desenvolvida com o Streamlit, tornando a aplicação intuitiva e interativa.

## Funcionalidades

- Pré-processamento de textos (remoção de stopwords, lemmatização)
- Vetorização usando TF-IDF
- Classificação via modelo treinado
- Visualização dos principais termos e métricas de predição
- Upload de novos textos para classificação com Streamlit

## Tecnologias utilizadas

- Python 3.11+
- Streamlit
- Scikit-learn
- NLTK
- Pandas

## Como utilizar?
Caso queira utilizar ele pronto acesse o link do streamlit:
https://reviewsentimentai-kgj9tzw6hufqgurdl7ayg6.streamlit.app/

Para pré-processar o dataset e treinar o modelo 

1- Instale as bibliotecas necessarias que estão no requirements.txt
2- Coloque todos os scripts na mesma pasta
3- Coloque o dataset na mesma pasta dos scripts
4- Execute primeiro o limpeza_dataset.py, depois o treino_modelo.py
5- Agora se divirta!
