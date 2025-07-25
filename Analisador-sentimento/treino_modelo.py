import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    with open('colunas.pkl', 'rb') as arquivo:
        dataset = pickle.load(arquivo)
except FileNotFoundError:
    print('Arquivo não encotrado')

def vetorizando_e_separando(dataset):
    # transformando texto em números
    vectorizer = TfidfVectorizer(max_features=7000)
    # aplicando a vetorização nos nossos textos
    X = vectorizer.fit_transform(dataset['review_limpo']).toarray()
    # colocando o atributo meta que contem as classes (positivo e negativo) no Y
    y = dataset['polarity']
    return X, y, vectorizer

def treinando_modelo(X, y):
    # dividindo as bases em treinamento e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    # inicializando o classifiacador naive bayes
    modelo = MultinomialNB()
    # treinando o modelo
    modelo.fit(X_treino, y_treino)
    return  modelo, X_teste, y_teste

def graficos(modelo,X_teste, y_teste):
    # fazendo as previsoes
    previsao = modelo.predict(X_teste)
    print(confusion_matrix(y_teste, previsao)) # matriz de confusão
    print(classification_report(y_teste, previsao)) # gera um relatório da precisão do modelo
    print(accuracy_score(y_teste, previsao)) # porcentagem de acerto do modelo

def função_principal():
    X, y, vectorizer = vetorizando_e_separando(dataset)
    modelo_treinado = treinando_modelo(X, y)
    modelo, X_teste, y_teste = modelo_treinado
    graficos(modelo, X_teste, y_teste)
    return modelo, vectorizer

if __name__ == '__main__':
    modelo, vectorizer = função_principal()

    with open('modelo.pkl', 'wb') as arquivo:
        pickle.dump(modelo, arquivo)

    with open('vetorizador.pkl', 'wb') as arquivo2:
        pickle.dump(vectorizer, arquivo2)
