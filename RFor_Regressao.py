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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#Criação do modelo
modelo = RandomForestRegressor(criterion= 'mse')
#Parâmetros a serem variados
parameter_space = {
    'min_samples_split': [2,4,8,10,100],
    'min_samples_leaf' : [1,2,4,8,10,100],
    'max_features': ['auto', 'sqrt', 'log2'],
}
#Criação do modelo de pesquisa dos parâmetros
model = GridSearchCV(modelo, parameter_space, n_jobs=-1, cv=5, verbose =1)
model.fit(X_train,Y_train) #treinamento do modelo
Y_pred = model.predict(X_test) #Utilização do modelo para predizer os dados de teste
Y_pred_train=model.predict(X_train) #Utilização do modelo para predizer os dados de treino

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Exibindo as métricas
print("="*15)
print(" R² = %.4f" % (r2_score(Y_test,Y_pred)))
print("="*15)
print(" MAE = %.4f" % (mean_absolute_error(Y_test,Y_pred)))
print("="*15)
print(" MSE = %.4f" % (mean_squared_error(Y_test,Y_pred)))
print("="*15)
#Desnormalizando os dados
y_pred_i = scaler.inverse_transform(Y_pred)
y_test_i = scaler.inverse_transform(Y_test)
#Exibindo os melhores parâmetros 
print(model.best_params_)

import matplotlib.pyplot as plt
#Plotagem dos gráficos
plt.figure()
plt.plot([min(y_test_i),max(y_test_i)],[min(y_test_i),max(y_test_i)],'-r')
plt.scatter(y_pred_i,y_test_i)
plt.xlabel("Valor previsto")
plt.ylabel("Valor real")
plt.show()

#Salvando o Modelo
import joblib
joblib.dump(model, 'Kleber_RForR.pkl', compress=9)