# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:06:36 2025

@author: Jesus Coss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import statsmodels.api as sm

from src.dynamic_pricing_data_loader import cargar_y_preparar_datos

import os
import pandas as pd

def GetData():
    """
    Procesa y agrega datos de ventas y boletos para su análisis.

    Esta función extrae datos relevantes de un marco de datos cargado,
    los filtra para incluir solo ventas positivas y luego los agrega
    por fecha de corrida para obtener el total de boletos vendidos y
    el valor total de ventas. Los DataFrames resultantes se preparan
    para series de tiempo, excluyendo la última fila y convirtiendo
    el índice a formato de fecha y hora.

    Returns:
        tuple: Una tupla que contiene dos objetos pandas.DataFrame.
            - df_Bol (pandas.DataFrame): Un DataFrame con el conteo
              agregado de boletos vendidos por fecha.
            - df_Venta (pandas.DataFrame): Un DataFrame con el valor
              agregado de ventas por fecha.
    """
    # 1. Definición de rutas de archivos
    # Obtiene la ruta del directorio de trabajo actual.
    ruta_principal = os.getcwd()
    # Construye la ruta completa al archivo de configuración 'config.json'.
    config_path = os.path.join(ruta_principal, "config", "config.json")
    
    # 2. Carga y preparación inicial de datos
    # Carga y procesa los datos brutos a partir de la configuración.
    Frame = cargar_y_preparar_datos(config_path, ruta_principal)
    # Selecciona un subconjunto de columnas clave para el procesamiento posterior.
    Df = Frame[['FECHA_CORRIDA', 'VENTA', 'BOLETOS_VEND', 'TIPO_CORRIDA']]
    # Filtra el DataFrame para retener solo las filas con valores de 'VENTA' positivos.
    Df = Df[Df['VENTA'] > 0]
    # Df = Df[Df['TIPO_CORRIDA'] == 'NORMAL'] # Esta línea está comentada.
    
    # 3. Agregación y procesamiento del DataFrame de boletos
    # Agrupa los datos por fecha y calcula la suma de boletos. Se excluye la última fila.
    df_Bol = Df.groupby('FECHA_CORRIDA')['BOLETOS_VEND'].sum().reset_index().iloc[:-1, :]
    # Establece la columna de fecha como el índice del DataFrame.
    df_Bol = df_Bol.set_index(df_Bol.columns[0])
    # Asigna un nombre más descriptivo al índice.
    df_Bol.index.name = 'Fecha'
    # Convierte el índice a objetos de fecha y hora para facilitar el análisis de series de tiempo.
    df_Bol.index = pd.to_datetime(df_Bol.index)
    
    # 4. Agregación y procesamiento del DataFrame de ventas
    # Agrupa los datos por fecha y calcula la suma total de ventas. Se excluye la última fila.
    df_Venta = Df.groupby('FECHA_CORRIDA')['VENTA'].sum().reset_index().iloc[:-1, :]
    # Establece la columna de fecha como el índice del DataFrame.
    df_Venta = df_Venta.set_index(df_Venta.columns[0])
    # Asigna un nombre más descriptivo al índice.
    df_Venta.index.name = 'Fecha'
    # Convierte el índice a objetos de fecha y hora.
    df_Venta.index = pd.to_datetime(df_Venta.index)

    # 5. Retorno de resultados
    # Devuelve los dos DataFrames procesados.
    return df_Bol, df_Venta


