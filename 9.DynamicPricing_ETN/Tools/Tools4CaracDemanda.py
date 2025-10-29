# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:12:41 2025

@author: Jesus Coss
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import matplotlib.pyplot as plt



def clasificar_dia_hora(df, dia_col="NOMBRE_DIA_CORRIDA", hora_col="HORA_DECIMAL", col2var="BOLETOS_VEND"):
    """
    Clasifica por día y por hora en 'Alto' (1) o 'Bajo' (0) según proporciones acumuladas
    del total de la variable `col2var` (por ejemplo, boletos vendidos).

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las columnas de día, hora y variable base (por ejemplo BOLETOS_VEND).
    dia_col : str
        Nombre de la columna del día (ej. 'NOMBRE_DIA_CORRIDA').
    hora_col : str
        Nombre de la columna de la hora (ej. 'HORA_DECIMAL').
    col2var : str
        Nombre de la columna usada para calcular proporciones (ej. 'BOLETOS_VEND').

    Retorna:
    --------
    df : pd.DataFrame
        Mismo dataframe con nuevas columnas:
        - 'Clasif_dia_venta'
        - 'Clasif_hora_venta'
        - 'Clasif_dia_venta_code'
        - 'Clasif_hora_venta_code'
    """

    # -----------------------------
    # Clasificación por DÍA
    # -----------------------------
    # Agrupamos por día y sumamos la variable base
    resumen_dia = df.groupby(dia_col)[col2var].sum().reset_index(name="total_var")
    resumen_dia["prop"] = resumen_dia["total_var"] / resumen_dia["total_var"].sum()
    resumen_dia = resumen_dia.sort_values("prop", ascending=False)
    resumen_dia["prop_acum"] = resumen_dia["prop"].cumsum()

    # Clasificación Alto/Bajo 60/40
    resumen_dia["Clasif_dia_venta"] = resumen_dia["prop_acum"].apply(lambda x: "Alto" if x <= 0.6 else "Bajo")
    resumen_dia["Clasif_dia_venta_code"] = resumen_dia["Clasif_dia_venta"].map({"Alto": 1, "Bajo": 0})
    # 'texto' sería la columna que quieres dividir
    texto = resumen_dia['NOMBRE_DIA_CORRIDA']
    
    # Aplicar .str.split('_') para dividir cada elemento de la Serie
    # El método expand=True separa las partes en columnas distintas.
    partes_divididas = texto.str.split('_', expand=True)
    
    # Asignar las partes a nuevas columnas del DataFrame
    resumen_dia['Numero'] = partes_divididas[0]
    resumen_dia['Dia'] = partes_divididas[1]
    # -----------------------------
    # Clasificación por HORA
    # -----------------------------
    resumen_hora = df.groupby(hora_col)[col2var].sum().reset_index(name="total_var")
    resumen_hora["prop"] = resumen_hora["total_var"] / resumen_hora["total_var"].sum()
    resumen_hora = resumen_hora.sort_values("prop", ascending=False)
    resumen_hora["prop_acum"] = resumen_hora["prop"].cumsum()

    resumen_hora["Clasif_hora_venta"] = resumen_hora["prop_acum"].apply(lambda x: "Alto" if x <= 0.7 else "Bajo")
    resumen_hora["Clasif_hora_venta_code"] = resumen_hora["Clasif_hora_venta"].map({"Alto": 1, "Bajo": 0})

    HoraBuena= resumen_hora[resumen_hora["Clasif_hora_venta_code"] == 1]['HORA_DECIMAL']
    DiaBueno= resumen_dia[resumen_dia["Clasif_dia_venta_code"] == 1]['Numero']
    return HoraBuena, DiaBueno


def clasificar_meses(df, col_pax="PAX_SUBEN", anio_col="AÑO", mes_col="MES"):
    """
    Clasifica los meses en 'Alto' (1) o 'Bajo' (0) según su proporción acumulada
    de pasajeros ponderados por año (más peso al más reciente).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Debe contener columnas [AÑO, MES, PAX_SUBEN].
    col_pax : str
        Nombre de la columna con pasajeros subidos.
    anio_col : str
        Nombre de la columna de año.
    mes_col : str
        Nombre de la columna de mes.
    
    Retorna:
    --------
    df_result : pd.DataFrame
        Mismo dataframe de entrada con columnas `Clasif_mes` (Alto/Bajo)
        y `Clasif_mes_code` (1/0) agregadas.
    """

    # 1️ Calcular pesos por año (más reciente = mayor peso)
    anios = sorted(df[anio_col].unique(), reverse=True)
    pesos = {anio: 1/(i+1) for i, anio in enumerate(anios)}
    total = sum(pesos.values())
    pesos = {k: v/total for k, v in pesos.items()}  # normalizamos

    # 2️ Calcular PAX ponderado por año y mes
    df_pond = (
        df.groupby([anio_col, mes_col])[col_pax].sum().reset_index()
    )
    df_pond["peso"] = df_pond[anio_col].map(pesos)
    df_pond["pond"] = df_pond[col_pax] * df_pond["peso"]

    # 3️ Sumar ponderado por mes
    resumen = (
        df_pond.groupby(mes_col)["pond"].sum().reset_index()
    )

    # 4️ Calcular proporciones y proporción acumulada
    resumen["prop_mes"] = resumen["pond"] / resumen["pond"].sum()
    resumen = resumen.sort_values("prop_mes", ascending=False)
    resumen["prop_acum"] = resumen["prop_mes"].cumsum()

    # 5️ Clasificación por proporción acumulada
    resumen["Clasif_mes"] = resumen["prop_acum"].apply(lambda x: "Alto" if x <= 0.6 else "Bajo")
    resumen["Clasif_mes_code"] = resumen["Clasif_mes"].map({"Alto": 1, "Bajo": 0})

    MesBueno= resumen[resumen["Clasif_mes_code"] == 1]["MES"]

    return MesBueno
        
def calcular_kpis(df:pd.DataFrame) -> dict:
    kpis = pd.DataFrame()

    df = df[ df['DIAS_ANTICIPACION'] >= 0 ]
    df["DIAS_ANTICIPACION"] = df["DIAS_ANTICIPACION"].fillna(0)
    df = df[ df['VENTA'] > 0 ]
    df["NUM_ASIENTO"] = df["NUM_ASIENTO"].astype(int)

        
    df = df[ df['DIAS_ANTICIPACION'] >= 0 ]
    df = df[ df['VENTA'] > 0 ]
    df["NUM_ASIENTO"] = df["NUM_ASIENTO"].astype(int)


    ##################################################################
    # --- Top asientos más vendidos por hora y día
    kpis = (
        df.groupby(["CAPACIDAD_ASIENTOS_TRAMO", "NUM_ASIENTO"]).agg(
            DIAS_VENTA = ( "FECHA_OPERACION", "nunique" ),
            BOLETOS_VEND = ( "BOLETOS_VEND", "sum" ), 
            VENTA = ( "VENTA", "sum" ),
            PAX_SUBEN = ( "PAX_SUBEN", "sum" ),
            VENTA_PROM = ( "VENTA", "mean" ),
            VEL_VENTA_PROM = ( "DIAS_ANTICIPACION", "mean" ),
            TARIFA_BASE_PROM = ( "TARIFA_BASE_TRAMO", "mean" )
            ).sort_values( 
                by=["NUM_ASIENTO", "BOLETOS_VEND"], 
                ascending=[True, False] 
                ).reset_index()
    )
    kpis["PAX_X_DIA_PROM"] = kpis["PAX_SUBEN"] / kpis["DIAS_VENTA"]

    # Calcular total de boletos por cada capacidad_asientos_tramo
    kpis = kpis.sort_values(by=[ "CAPACIDAD_ASIENTOS_TRAMO", "PAX_X_DIA_PROM", "NUM_ASIENTO" ], ascending=[False, False, True] )
    # Paso 1: calcular el total de PAX_X_DIA_PROM por cada capacidad
    kpis["TOTAL_PAX_X_CAP"] = kpis.groupby("CAPACIDAD_ASIENTOS_TRAMO")["PAX_X_DIA_PROM"].transform("sum")
    # Paso 2: calcular la proporción por asiento
    kpis["PROP_ASIENTO"] = kpis["PAX_X_DIA_PROM"] / kpis["TOTAL_PAX_X_CAP"]
    # Paso 3: ordenar y calcular acumulado dentro de cada capacidad
    kpis["PROP_ACUM"] = kpis.groupby("CAPACIDAD_ASIENTOS_TRAMO")["PROP_ASIENTO"].cumsum()
    # OPCIONAL: convertir a porcentaje (0–100 en lugar de 0–1)
    kpis["PROP_ASIENTO_PCT"] = kpis["PROP_ASIENTO"] * 100
    kpis["PROP_ACUM_PCT"] = kpis["PROP_ACUM"] * 100

    kpis = kpis.reset_index(drop=True)


    return kpis

def BuenAsiento(Frame):
    kpi= calcular_kpis(Frame)

    Capacidad= kpi['CAPACIDAD_ASIENTOS_TRAMO'].unique()

    #"""
    Data={}
    for cap in Capacidad:
        # Definimos el valor de la capacidad que quieres filtrar
        capacidad_objetivo = cap
        
        # Aplicamos los dos filtros usando el operador lógico '&' (AND)
        filtro = (kpi['CAPACIDAD_ASIENTOS_TRAMO'] == capacidad_objetivo) & (kpi['PROP_ACUM'] <= 0.60)
        
        # Aplicamos el filtro al DataFrame
        resultado = kpi[filtro]['NUM_ASIENTO']
        Data[int(cap)]= list(resultado)
        
    return Data


def BuenasCaracteristicas(Frame):
    ruta_principal = os.getcwd()
    config_path = os.path.join(ruta_principal, "Files", "BuenaCaracteristicas.json")
    
        
    HoraBuena, DiaBueno = clasificar_dia_hora(Frame.copy())
    HoraBuena= list(HoraBuena)
    DiaBueno= list(map(int, list(DiaBueno)))
    MesBueno= clasificar_meses(Frame.copy())
    MesBueno= list(MesBueno)
    AsientosBuenos= BuenAsiento(Frame)
    
    data={"MesBueno": MesBueno, "HoraBuena":HoraBuena, "DiaBueno": DiaBueno,
          "AsientosBuenos":AsientosBuenos
        }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        