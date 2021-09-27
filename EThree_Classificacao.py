import xlrd
# Importando os dados  a partir da planiha
loc = (r'C:\Users\LASSOP_2\Desktop\VSCODECODES\Kleber_SVM\Geracao_Dados.xlsx') #Caminho da planilha em formato xlsx
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
nl = int(sheet.nrows) #n° linhas da planilha
nc = int(sheet.ncols) #n° colunas da planilha

X = []
Y = []
#Atribuindo os dados de entrada e saída as listas X e Y respectivamente.
for i in range(nc):
    L = []
    for j in range(1,nl):
        L.append(float(sheet.cell_value(j,i)))
        if j == 1:
            print(f'Coletando a Linha {sheet.cell_value(0,i)} coluna = {i}')
    if i <3:
        X.append(L)
    elif i == 3:
        Y.append(L)


import numpy as np
#Transformando as listas em array e redimensionalizando
Y = np.array(Y)
X = np.array(X)
print(X.shape)
print(Y.shape)
X = np.transpose(X)
Y = np.transpose(Y)
Y = Y.reshape(2700,1)
print(X.shape)
print(Y.shape)

#Normalizando os dados
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

Y = np.squeeze(Y)

from sklearn.model_selection import train_test_split
#Separando os dados para treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
#Criação do modelo de Classificação
modelo = ExtraTreesClassifier(random_state=42)
#Parâmetros a serem variados
parameter_space = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [100,200,300,500,900],
    'min_samples_split': [2,4,8,10,100],
    'min_samples_leaf' : [1,2,4,8,10,100],
    'max_features': ['auto', 'sqrt', 'log2'],

}
#Criação do modelo de pesquisa dos parâmetros
model = GridSearchCV(modelo, parameter_space, n_jobs=-1, cv=5, verbose =1)
model.fit(X_train,Y_train) #treinamento do modelo
Y_pred = model.predict(X_test) #Utilização do modelo para predizer os dados de teste
Y_pred_train=model.predict(X_train) #Utilização do modelo para predizer os dados de treino

# Exibindo as métricas
print(model.best_params_)

import matplotlib.pyplot as plt

import pandas as pd
#Plotagem das matrizes de confusão
df_confusion = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)

import matplotlib.pyplot as plt

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Plotagem das métricas de classificação
print(classification_report(Y_pred,Y_test)) 
print(confusion_matrix(Y_pred,Y_test))
print(df_confusion)
plot_confusion_matrix(df_confusion)