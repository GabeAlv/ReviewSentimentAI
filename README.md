# 🧠 Análise de Sentimentos com NLP e Naive Bayes

Este projeto aplica técnicas de Processamento de Linguagem Natural (NLP) para realizar uma análise de sentimentos em avaliações de clientes utilizando o algoritmo Naive Bayes e vetorização com TF-IDF. O conjunto de dados utilizado é da empresa B2W Digital.

## 📌 Objetivo

O objetivo principal é classificar automaticamente os sentimentos (positivos ou negativos) expressos em comentários de clientes, utilizando um modelo supervisionado de aprendizado de máquina.

---

## 📊 Dataset

Utilizamos o dataset da **B2W Digital**, que contém mais de 120000 avaliações de clientes sobre produtos. As colunas utlizadas foram:

- `review_text` – as reviews dos clientes
- `rating` – a nota atribuída pelo cliente
- `polarity` – a classe de sentimento 


## 🛠️ Tecnologias Utilizadas

- Python 3.10+
- Pandas
- Scikit-learn
- NLTK
- Spacy
- joblib
- Streamlit

---

## ⚙️ Pré-processamento (NLP)

- Remoção de pontuação e stopwords
- Tokenização
- Lemmatização
- Vetorização com **TF-IDF**

---

## 🧪 Algoritmo Utilizado

O modelo de **Naive Bayes (MultinomialNB)** foi utilizado por sua eficácia e simplicidade em tarefas de classificação de texto.

---

## 📈 Resultados
Acurácia: 87%

Precisão, recall e f1-score foram analisados.

| Classe      | Precisão | Revocação | F1-Score | Suporte |
|-------------|----------|-----------|----------|---------|
| Negativo    | 0.85     | 0.73      | 0.79     | 7.128   |
| Positivo    | 0.89     | 0.94      | 0.92     | 16.084  |
| **Média**   | **0.88** | **0.88**  | **0.88** | 23.212  |

Matriz de confusão foi analisada
[[ 5198  1930]
 [  885 15199]]
 
 ---
 
## 🐍 Como utilizar?
Caso queira utilizar ele pronto acesse o link do streamlit:
https://reviewsentimentai-kgj9tzw6hufqgurdl7ayg6.streamlit.app/

Para pré-processar o dataset e treinar o modelo 
1. Instale as bibliotecas necessarias que estão no requirements.txt
2. Coloque todos os scripts na mesma pasta
3. Coloque o dataset na mesma pasta dos scripts
4. Execute primeiro o limpeza_dataset.py, depois o treino_modelo.py
5. Agora se divirta!
