# OS PRINTS FORAM COMENTADOS PARA EVITAR SPAM NO TERMINAL, 
# DESCOMENTE-OS AOS POUCOS PARA VER OS RESULTADOS
# Aluno: André Arantes Lopes
# Matrícula: 779447

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
from sklearn import tree

# Importando a base da dados
base = pd.read_csv('content/restaurante.csv', sep=';')

# Contando quantas opções de resposta tem cada atributo
# print(np.unique(base['Conc'], return_counts=True))
 
# Construindo as tabelas de atributos de entrada e atributos de classificação 
X = base.iloc[:, 0:10].values
Y = base.iloc[:, 10].values

# print("Atributos de entrada:")
# print(X)
# print("Atributos de classificação:")
# print(Y)

# Função criada para transformar os atributos de entrada qualitativos ordinais em numéricos 
def transform_with_labelEncoder(df, col:int):
  df[:, col] = LabelEncoder().fit_transform(df[:, col])

# Cópia da base x para alterar os valores
X_encoded = X.copy()

transform_with_labelEncoder(X_encoded, 0);
transform_with_labelEncoder(X_encoded, 1);
transform_with_labelEncoder(X_encoded, 2);
transform_with_labelEncoder(X_encoded, 3);
transform_with_labelEncoder(X_encoded, 4);
transform_with_labelEncoder(X_encoded, 5);
transform_with_labelEncoder(X_encoded, 6);
transform_with_labelEncoder(X_encoded, 7);
transform_with_labelEncoder(X_encoded, 9);

# Uso do OneHotEncoder para transformar os atributos qualitativos nominais em numéricos
one_hot_encoder_transformer = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')
X_encoded = one_hot_encoder_transformer.fit_transform(X_encoded)

# print(X_encoded);
# print(X_encoded.shape);


# Divisão dos dados em treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(
  X_encoded,
  Y,
  test_size = 0.20,
  random_state = 23
)

# Implementação do modelo de árvore de decisão
modelo = DecisionTreeClassifier(criterion='entropy')
treinando = modelo.fit(X_treino, Y_treino)

# Teste
previsoes = modelo.predict(X_teste)
# print("Acuracia do modelo:", accuracy_score(Y_teste, previsoes))

# Construção da matriz de confusão
# print(confusion_matrix(Y_teste, previsoes))

cm = ConfusionMatrix(modelo)
cm.fit(X_treino, Y_treino)
cm.score(X_teste, Y_teste)

# Imprimindo a matriz de confusão
plt.figure(figsize=(8, 13))
plt.show()

# Métricas do modelo
# print(classification_report(Y_teste, previsoes))

# Geração da árvore
previsores = ['Frances','Hamburguer','Italiano','Tailandes','Alternativo','Bar','SexSab','Fome','Cliente','Preço','Chuva','Res','Tipo','Tempo']
tree.plot_tree(
  modelo, 
  feature_names=previsores, 
  class_names=modelo.classes_.tolist(), 
  filled=True, 
  )

# A árvore gerada estará disponível em um arquivo chamado "decision_tree.png"
plt.savefig('decision_tree.png')
