# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:37:03 2025

@author: Jesus Coss
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json
from xgboost import XGBClassifier # Importamos el clasificador de XGBoost
import joblib

def GetTodayData4Cluster(Frame):
    Df=Frame.copy()
    
    Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
    fecha_maxima = Df['FECHA_OPERACION'].max()
    Df = Df[Df['FECHA_OPERACION'] == fecha_maxima].copy()
    
    # Filtrar registros con VENTA_TOTAL > 0 (elimina ventas nulas o negativas).
    Df=Df[Df[ 'VENTA_TOTAL']>0]
    Df=Df[Df[ 'BOLETOS_VEND']>0]
    Df['HORAS_ANTICIPACION']=Df['HORAS_ANTICIPACION'].abs()
    Df['DIAS_ANTICIPACION']=Df['DIAS_ANTICIPACION'].abs()

    # Rellenar valores faltantes en PORCENT_PROMO con 0.
    Df['PORCENT_PROMO'] = Df['PORCENT_PROMO'].fillna(0)

    # Construir un DataFrame 'known' con nombres y emails conocidos:
    # - eliminar filas sin EMAIL,
    # - eliminar duplicados por NOMBRE_PASAJERO,
    # - quedarnos solo con ['NOMBRE_PASAJERO', 'EMAIL'].
    known = (
        Df
        .dropna(subset=['EMAIL'])
        .drop_duplicates(subset=['NOMBRE_PASAJERO'])
        [['NOMBRE_PASAJERO','EMAIL']]
    )

    # Hacer un merge para anotar cada fila con el EMAIL conocido (si existe).
    # Se crea temporalmente la columna 'EMAIL_KNOWN' para contener el email mapeado.
    Df = Df.merge(
        known,
        on='NOMBRE_PASAJERO',
        how='left',
        suffixes=('','_KNOWN')
    )

    # Rellenar la columna EMAIL con EMAIL_KNOWN cuando EMAIL original esté vacío.
    # Esto prioriza el email original; si es NaN, toma el conocido.
    Df['EMAIL'] = Df['EMAIL'].fillna(Df['EMAIL_KNOWN'])

    # Eliminar la columna auxiliar EMAIL_KNOWN que ya no se necesita.
    Df.drop(columns=['EMAIL_KNOWN'], inplace=True)

    # Dominio que se usará para generar emails genéricos (placeholder).
    dominio = 'ejemplo.com'

    # Obtener la lista de nombres únicos sin email asignado.
    nombres_sin_email = Df.loc[Df['EMAIL'].isna(), 'NOMBRE_PASAJERO'].unique()

    # Crear un diccionario mapping nombre -> email genérico (ej: "juan.perez@ejemplo.com").
    # Nota: se hace una transformación simple (lower + reemplazar espacios por puntos).
    generic_map = {
        nombre: f"{nombre.lower().replace(' ','.')}@{dominio}"
        for nombre in nombres_sin_email
    }

    # Asegurar que la columna EMAIL es de tipo object (cadena).
    Df['EMAIL'] = Df['EMAIL'].astype('object')

    # Rellenar los emails faltantes usando el mapping generado.
    # Se usa map sobre NOMBRE_PASAJERO y fillna para no sobrescribir emails existentes.
    Df['EMAIL'] = Df['EMAIL'].fillna(Df['NOMBRE_PASAJERO'].map(generic_map))
    
    # Corrección de categorías:
    # Si un adulto tiene promoción > 0, renombrar su descuento como "PROMOCION ESPECIAL"
    Df.loc[(Df['PORCENT_PROMO'] > 0) & (Df['DESC_DESCUENTO'] == 'ADULTO'), 'DESC_DESCUENTO'] = 'PROMOCION ESPECIAL'
    
    # Si tiene 0% de promoción pero estaba marcado como "PROMOCION ESPECIAL", devolverlo a "ADULTO"
    Df.loc[(Df['PORCENT_PROMO'] == 0) & (Df['DESC_DESCUENTO'] == 'PROMOCION ESPECIAL'), 'DESC_DESCUENTO'] = 'ADULTO'
    
    Df['VENTA_ANTICIPADA']= np.where(Df['VENTA_ANTICIPADA']=='SI',1,0)
    # Devolver el DataFrame final procesado.
    return Df

def GetDescuento(Cluster):
    if Cluster== 0 or Cluster== 2:
        return 0.02
    elif Cluster== 3 or Cluster== 5:
        return -0.02
    elif Cluster== 1 or Cluster== 4:
        return 0.0
    
# En esta funcion se busca si el cliente ya está en la base de datos o no
def GetCluster(Df,DB):
    if Df['EMAIL'].isin(DB['EMAIL']).iloc[0]: 
        print("En la lista")
        Cluster=DB[DB['EMAIL']==Df['EMAIL'].iloc[0]]['Cluster'].iloc[0]
        desc=GetDescuento(Cluster)
        return Cluster,desc
    else:
        print("Fuera de la lista")
        columnas_destino= DB.columns[1:-1]
        df_ = pd.DataFrame(
        0,                                # Valor a rellenar (cero)
        index=Df.index,                   # Usamos el índice de DB para la altura
        columns=columnas_destino          # Usamos las columnas seleccionadas
        )
    
        columnas_origen = [
            'BOLETOS_VEND',
            'VENTA', # Primera columna VENTA
            'VENTA', # Segunda columna VENTA
            'PORCENT_PROMO',
            'HORAS_ANTICIPACION',
            'VENTA_ANTICIPADA'
        ]
        
        df_.loc[:, df_.columns[:6]] = Df[columnas_origen].values
        df_['Recencia']=0
        df_dummies_desc = pd.get_dummies(Df['DESC_DESCUENTO'], prefix='DESC').astype(int)
        df_dummies_pago = pd.get_dummies(Df['PAGO_METODO'], prefix='PAGO').astype(int)
    
        df_[df_dummies_desc.columns[0]]=df_dummies_desc[df_dummies_desc.columns[0]].iloc[0]
        df_[df_dummies_pago.columns[0]]=df_dummies_pago[df_dummies_pago.columns[0]].iloc[0]
    
        modelo_cargado = XGBClassifier()
        # Cargar el modelo guardado
        modelo_cargado.load_model("modelo_xgboost_clientes.json")
    
        Cluster = modelo_cargado.predict(df_)[0]
        desc=GetDescuento(Cluster)
        return Cluster,desc