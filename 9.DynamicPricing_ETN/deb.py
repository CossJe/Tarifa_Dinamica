# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 13:28:22 2025

@author: Jesus Coss
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import statsmodels.formula.api as smf

from src.dynamic_pricing_data_loader import cargar_y_preparar_datos

def Get_Data():
    # Obtener el directorio de trabajo actual (ruta principal del proyecto).
    ruta_principal = os.getcwd()

    # Construir la ruta al archivo de configuración "config/config.json".
    config_path = os.path.join(ruta_principal, "config", "config.json")

    # Llamar a la función externa que carga y realiza preprocesamiento inicial.
    Frame = cargar_y_preparar_datos(config_path, ruta_principal)
    
    # Seleccionar solo las columnas relevantes para el análisis.
    D4NN = Frame[["PAX_SUBEN", 'FECHA_OPERACION','VENTA','ORIGEN', 'DESTINO','BOLETOS_VEND','TARIFA_BASE_TRAMO',
                 'HORA_SALIDA_CORRIDA']].copy()
    D4NN= D4NN[(D4NN['ORIGEN']=='MEXN') & (D4NN['DESTINO']== 'GDLJ')].copy()
    D4NN=D4NN.iloc[:-1]
    return D4NN

D4NN= Get_Data()