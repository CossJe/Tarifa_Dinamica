# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:06:36 2025

@author: Jesus Coss
"""

import matplotlib.pyplot as plt
import json

import DB
import Entrenar1Var as Ent1
import Entrenar2Var as Ent2
import Predecir1Var as Pred1
import Predecir2Var as Pred2

def MainOfer():
    """
    Función principal para el pronóstico de ventas y boletos.

    Esta función orquesta un pipeline completo de pronóstico que incluye:
    - Carga de datos de ventas y boletos.
    - Pronóstico de la serie de ventas de forma independiente.
    - Pronóstico de la serie de boletos utilizando las predicciones de ventas
      como una variable predictora.
    - Visualización del resultado final.

    Retorna:
    --------
    tuple
        - pandas.DataFrame: DataFrame con las predicciones de boletos.
        - datetime.Timestamp: La última fecha del DataFrame original de boletos.
    """
    # Se define el identificador de la empresa.
    Emp = 'ETN'
    
    # Se obtienen los datos de la base de datos.
    df_Bol, df_Venta = DB.GetData()
    
    # Se define el número de días a pronosticar.
    PronosDias = 15
    
    # Pronóstico de la variable 'Venta'.
    # Se realiza una búsqueda de hiperparámetros para encontrar el mejor modelo.
    CaracVenta, Modelo_Xgb_Venta = Ent1.GridSearch(df_Venta, Emp, df_Venta.columns[0])
    # Se generan las predicciones de ventas para los próximos 15 días.
    df_Forecast_Venta = Pred1.predicciones(df_Venta, CaracVenta, Modelo_Xgb_Venta, PronosDias)
    
    # Pronóstico de la variable 'Boletos' utilizando las predicciones de 'Venta'.
    # Se realiza una búsqueda de hiperparámetros, pasando los datos de 'Venta'
    # como una variable adicional para el entrenamiento.
    CaracBol, Modelo_Xgb_Bol = Ent2.GridSearch(df_Bol, Emp, df_Bol.columns[0], df_Venta)
    # Se generan las predicciones de boletos. Se pasan las predicciones de ventas
    # ('df_Forecast_Venta') para ser utilizadas en la fase de pronóstico.
    df_Forecast_Bol, last_date = Pred2.predicciones(df_Bol, df_Forecast_Venta, CaracBol, Modelo_Xgb_Bol, PronosDias)
    
    # Se crea una figura para el gráfico de visualización.
    plt.figure(figsize=(15, 7))
    
    # Se grafican los valores históricos de boletos en negro.
    plt.plot(df_Bol.index, df_Bol.values, label='Valores Verdaderos', color='black', linestyle='-', linewidth=2)
    
    # Se grafican las predicciones de boletos en azul.
    plt.plot(df_Forecast_Bol.index, df_Forecast_Bol.values, label='Predicciones del Modelo', color='blue', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Se añaden títulos y etiquetas al gráfico.
    plt.title(f'Comparación de Valores Verdaderos vs. Predicciones para {Emp} de ' + str(df_Bol.columns[0]), fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel(df_Bol.columns[0], fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Se muestra el gráfico.
    plt.show()
    
    # Se retornan los resultados.
    return df_Forecast_Bol, last_date

df_Forecast_Bol, last_date= MainOfer()