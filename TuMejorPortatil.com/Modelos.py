import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#Limpieza de datos:

def limpiar_dataframe(archivo_csv):

    df = pd.read_csv(archivo_csv, index_col="id")
    
    df.sort_values(by="laptop_ID", inplace=True)

    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(float)

    return df

########df_limpio = limpiar_dataframe("train.csv")#########

   




#Dibujar el HeatMap de correlaciones:

def dibujar_heatmap(df, figsize=(10, 5), vmin=-1, vmax=1, cmap="coolwarm"):
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), vmin=vmin, vmax=vmax, cmap=cmap, annot=True)
    plt.show()


#Modelo lineal:

def modelo_regresion(train_df, X_columns, Y_column, test_size=0.2, random_state=40):

    X = train_df[X_columns]
    Y = train_df[Y_column]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, Y_pred)
    
    print("MAE en conjunto de validación:", mae)

    return model, scaler


#Modelo regresion:

def modelo_regresion_polinomica(train_df, X_columns, Y_column, degree=2, test_size=0.2, random_state=40):

    X = train_df[X_columns]
    Y = train_df[Y_column]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, Y_train)

    Y_pred = model.predict(X_test_poly)
    mae = mean_absolute_error(Y_test, Y_pred)

    print("MAE en conjunto de validación:", mae)

    return model, scaler, poly


#Probar modelo

def probar_modelo(model, scaler, test_df, X_columns, output_csv='predicciones.csv', poly=None):

    X_test_final = test_df[X_columns]
    X_test_final = scaler.transform(X_test_final)

    if poly is not None:
        X_test_final = poly.transform(X_test_final)

    Y_pred_test = model.predict(X_test_final)

    resultados = pd.DataFrame({'id': test_df.index, 'Price_euros': Y_pred_test})

    resultados.to_csv(output_csv, index=False)
    print(f"Predicciones guardadas en {output_csv}")






