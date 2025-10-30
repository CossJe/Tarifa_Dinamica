# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:44:47 2025

@author: Jesus Coss
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from src.dynamic_pricing_data_loader import cargar_y_preparar_datos

def Get_Data():
    # Obtener el directorio de trabajo actual (ruta principal del proyecto).
    ruta_principal = os.getcwd()

    # Construir la ruta al archivo de configuración "config/config.json".
    config_path = os.path.join(ruta_principal, "config", "config.json")

    # Llamar a la función externa que carga y realiza preprocesamiento inicial.
    Frame = cargar_y_preparar_datos(config_path, ruta_principal)

    # Seleccionar solo las columnas relevantes para el análisis.
    D4NN = Frame[['NOMBRE_PASAJERO','BOLETOS_VEND',"FECHA_CORRIDA", "HORA_SALIDA_CORRIDA", "CLASE_SERVICIO", 'IVA_TARIFA_BASE_TRAMO',
    "PAX_SUBEN", "TARIFA_BASE_TRAMO",'FECHA_OPERACION', 'HORA_OPERACION','VENTA','DISPONIBILIDAD_TRAMO',
    'HORAS_ANTICIPACION','ORIGEN', 'DESTINO','TIPO_CLIENTE','NUM_ASIENTO','CAPACIDAD_ASIENTOS_TRAMO'
                  ]].copy()

    D4C = Frame[['NOMBRE_PASAJERO','BOLETOS_VEND', 'CLASE_SERVICIO', 'DESC_DESCUENTO', 'DIAS_ANTICIPACION',
                'EMAIL', 'FECHA_CORRIDA', 'FECHA_OPERACION', 'HORAS_ANTICIPACION',
                'PAGO_METODO', 'PORCENT_PROMO',
                'TIPO_CORRIDA', 'TIPO_PASAJERO',  'TOTAL_BOLETOS',
                'VENTA', 'VENTA_ANTICIPADA', 'VENTA_TOTAL']]
    
    D4C1= D4C.copy()
    D4C1['TARIFA']= D4NN["TARIFA_BASE_TRAMO"] - D4NN["IVA_TARIFA_BASE_TRAMO"]
    
    #D4GC= Frame[['CAPACIDAD_ASIENTOS_TRAMO','FECHA_OPERACION',]]
    
    return D4NN, D4C, D4C1, Frame

def GetDB():
    ruta_principal = os.getcwd()
    config_path = os.path.join(ruta_principal, "Files", "ClusteringClientes_Clustering.parquet")
    DB = pd.read_parquet(config_path)
    return DB

def Get_Data4NN():
    # Obtener el directorio de trabajo actual (ruta principal del proyecto).
    ruta_principal = os.getcwd()

    # Construir la ruta al archivo de configuración "config/config.json".
    config_path = os.path.join(ruta_principal, "config", "config.json")

    # Llamar a la función externa que carga y realiza preprocesamiento inicial.
    Frame = cargar_y_preparar_datos(config_path, ruta_principal)
    columnas = [
    'FECHA_CORRIDA',
    'HORA_SALIDA_ORIGEN_CORRIDA',
    'TIPO_PASAJERO',
    'PAGO_METODO',
    'PORCENT_PROMO',
    'BOLETOS_VEND',
    'AÑO',
    'DIF_TARIF',
    'CLASE_SERVICIO',
    'ORIGEN',
    'DESTINO',
    'MES',
    'KMS_TRAMO',
    'OCUPACION_TRAMO',
    'CAPACIDAD_ASIENTOS_TRAMO',
    'HORAS_ANTICIPACION',
    'DISPONIBILIDAD_TRAMO',
    'NOMBRE_DIA_CORRIDA'
    ]
    
    D4NN= Frame[columnas].copy()
    
    return  D4NN
