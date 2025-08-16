# importando as bibliotecas
import pandas as pd # biblioteca para manipulação de dados
import re # biblioteca para expressões regulares
import nltk # biblioteca para nlp com ferramentas básicas de NLP (tokenização, stem, stopwords)
from nltk.corpus import stopwords
import spacy  # biblioteca de NLP robusto com NER e análise sintática
import pickle # biblioteca para baixar arquivos

#baixando as stopwords
nltk.download('stopwords')
#baixando o modelo de lematização
spacy.cli.download('pt_core_news_sm')

class pré_processamento:
    def __init__(self, data):
        self.data = data

    # removendo colunas e alterando valores
    def limpando_dataset(self):
      print('Limpando o dataset...')
    #tirando as colunas que nao vamos usar
      self.data = self.data.drop(columns=['original_index', 'review_text_processed', 'review_text_tokenized', 'kfold_polarity', 'kfold_rating'])
      self.data['polarity'] = self.data['polarity'].astype('object')
      valores_nulos = self.data.loc[self.data['polarity'].isnull()] # localizando os registros com valores nulos
      self.data = self.data.drop(valores_nulos.index) # retirando os registros com valores nulos
    #alterando o nome das classes
      self.data.loc[self.data['polarity'] == 1, 'polarity'] = 'Positivo'
      self.data.loc[self.data['polarity'] == 0, 'polarity'] = 'Negativo'
      return self

    # método para limpar o texto
    def limpar_texto_dataset(self):
        print('Limpando os reviews...')
        self.data['review_limpo'] = self.data['review_text'].apply(self.limpar_texto) # aplicando a limpeza
        return self


    def limpar_texto(self, texto):
      texto = re.sub(r'@\w+', '', texto) # remove menções
      texto = re.sub(r'https?://\S+', '', texto) # remove URLs
      texto = re.sub(r'[^a-zA-ZÀ-ÿ]', ' ', texto) # remove caracteres nao alfabéticos
      texto = texto.lower() # colocando os textos no minusculo
      return texto

    # tirando as stopwords
    def tirando_stopwords(self):
        print('Retirando stopwords...')
        #pegando as stopword do português
        stop_words = set(stopwords.words('portuguese'))
        self.data['review_limpo'] = self.data['review_limpo'].apply(lambda x: ' '.join([palavra for palavra in x.split() if palavra not in stop_words])) #aplicando a limpeza
        return self

    # lematizando os textos (colocando as palavras em sua forma base)
    def lematizar(self, nlp):
        print('lematizando...')
        resultados = []
        #processando os textos em lote com o spacy para melhor eficiência
        for doc in nlp.pipe(self.data['review_limpo'].tolist(), batch_size=2000):  # batch_size controla quantos textos são processados por vez
            # Extrai os lemas (formas base) apenas de substantivos e verbos
            texto_lematizado = ' '.join([
                token.lemma_  # Pega o lema da palavra
                for token in doc  # Para cada token no documento
                if token.pos_ in ['NOUN', 'VERB']  # Filtra só substantivos e verbos
            ])
            resultados.append(texto_lematizado) # Adiciona o texto lematizado à lista de resultados
        self.data['review_limpo'] = resultados
        return self.data

nlp = spacy.load('pt_core_news_sm') # carregando o modelo de lematização
data = None
try:
    data = pd.read_csv('b2w.csv') # pegando o dataset que vamos usar
except FileNotFoundError:
    print('Arquivo não encontrado')

if __name__ == '__main__':
    pre = pré_processamento(data) # criando uma instância da classe
    dataset_final = pre.limpando_dataset().limpar_texto_dataset().tirando_stopwords().lematizar(nlp) # utilizando os métodos do objeto
    colunas_para_salvar = dataset_final[['review_limpo', 'polarity', 'rating']] # guardando as colunas necessárias
    print('Pré processamento concluído!!')

    # baixando as colunas que vao ser usadas posteriormente (coluna rating para visualização com gráficos)
    with open('colunas.pkl', 'wb') as arquivo:
        pickle.dump(colunas_para_salvar, arquivo)