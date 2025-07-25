# importando as bibliotecas
import pandas as pd # biblioteca para manipulação de dados
import re # biblioteca para expressões regulares
import nltk # biblioteca para nlp com ferramentas básicas de NLP (tokenização, stem, stopwords)
from nltk.corpus import stopwords
import spacy  # biblioteca de NLP robusto com NER e análise sintática
import pickle # biblioteca para baixar arquivos

#baixando as stopwords
nltk.download('stopwords')
#baixando e carregando o modelo de lematização
spacy.cli.download('pt_core_news_sm')
nlp = spacy.load('pt_core_news_sm')

try:
    #carregando o dataset
    data = pd.read_csv('b2w.csv')
except FileNotFoundError:
    print('Dataset não encontrado')

# removendo colunas e alterando valores
def limpando_dataset(data):
#tirando as colunas que nao vamos usar
  data = data.drop(
    columns=['original_index', 'review_text_processed', 'review_text_tokenized', 'kfold_polarity', 'kfold_rating'])
  data['polarity'] = data['polarity'].astype('object')
  valores_nulos = data.loc[data['polarity'].isnull()] # localizando os registros com valores nulos
  data = data.drop(valores_nulos.index) # retirando os registros com valores nulos
#alterando o nome das classes
  data.loc[data['polarity'] == 1, 'polarity'] = 'Positivo'
  data.loc[data['polarity'] == 0, 'polarity'] = 'Negativo'
  return data

# função para limpar os textos
def limpar_texto(texto):
  texto = re.sub(r'@\w+', '', texto) # remove menções
  texto = re.sub(r'https?://\S+', '', texto) # remove URLs
  texto = re.sub(r'[^a-zA-ZÀ-ÿ]', ' ', texto) # remove caracteres nao alfabéticos
  texto = texto.lower() # colocando os textos no minusculo
  return texto

# tirando as stopwords
def tirando_stopwords(data):
    #pegando as stopword do português
    stop_words = set(stopwords.words('portuguese'))
    data['review_limpo'] = data['review_limpo'].apply(lambda x: ' '.join([palavra for palavra in x.split() if palavra not in stop_words])) #aplicando a limpeza
    return data

# lematizando os textos (colocando as palavras em sua forma base)
def lematizar(lista_textos, nlp):
    resultados = []
    #processando os textos em lote com o spacy para melhor eficiência
    for doc in nlp.pipe(lista_textos, batch_size=2000):  # batch_size controla quantos textos são processados por vez
        # Extrai os lemas (formas base) apenas de substantivos e verbos
        texto_lematizado = ' '.join([
            token.lemma_  # Pega o lema da palavra
            for token in doc  # Para cada token no documento
            if token.pos_ in ['NOUN', 'VERB']  # Filtra só substantivos e verbos
        ])
        resultados.append(texto_lematizado) # Adiciona o texto lematizado à lista de resultados
    return resultados

def função_principal():
    print('limpando dataset...')
    dataset_limpo = limpando_dataset(data) # recebendo o dataset tratado
    dataset_limpo['review_limpo'] = dataset_limpo['review_text'].apply(limpar_texto) # aplicando a função de limpeza na coluna com os reviews
    print('tirando as stopwords...')
    dataset_limpo = tirando_stopwords(dataset_limpo) # retirando as stopwords dos reviews
    print('lematizando...')
    dataset_limpo['review_limpo'] = lematizar(dataset_limpo['review_limpo'].tolist(), nlp) # lematizando os reviews
    return dataset_limpo

if __name__ == '__main__':
    data_final = função_principal() # chamando a função principal e recebendo o dataset final
    colunas_para_salvar = data_final[['review_limpo', 'polarity', 'rating']]

    # baixando as colunas que vao ser usadas posteriormente (coluna rating para visualização com graficos)
    with open('colunas.pkl', 'wb') as arquivo:
        pickle.dump(colunas_para_salvar, arquivo)

