import pandas as pd
import numpy as np
import pyodbc
import os
import unicodedata

import warnings
warnings.filterwarnings('ignore')

#################################################################################################################
#### función que realizar la estimación de la venta por día de la semana y hora decimal
def ventas_x_dia_hora(df: pd.DataFrame) -> pd.DataFrame:    
    # --- Agrupación por día ---
    ventas_dia_hora = (
        df.groupby(["ORIGEN", "DESTINO", "AÑO", "NOMBRE_DIA_CORRIDA", "HORA_DECIMAL"])
        .agg({
            "BOLETOS_VEND": "sum",
            "VENTA": "sum",
            "OCUPACION_TRAMO": "mean",
            "CAPACIDAD_ASIENTOS_TRAMO": "max"
        })
        .reset_index()
    )
    
    return ventas_dia_hora


#################################################################################################################
#### función que calcula los KPIs para el EDA
def calcular_kpis(df:pd.DataFrame, colYear, valYear) -> dict:
    kpis = {}

    df2 = df.copy()
    if valYear != "ALL":
        df = df.loc[ df[colYear] >= valYear ]

    df = df[ df['DIAS_ANTICIPACION'] >= 0 ]
    df = df[ df['VENTA'] > 0 ]

    ##################################################################
    # --- Ocupación promedio por hora, día y origen
    kpis["ocupacion_promedio"] = (
        df.groupby(["NOMBRE_DIA_CORRIDA", "HORA_DECIMAL"])[["OCUPACION_TRAMO", "CAPACIDAD_ASIENTOS_TRAMO"]]
        .mean()
        .reset_index()
    )
    kpis["ocupacion_promedio"]["FOP_PROM"] = kpis["ocupacion_promedio"]["OCUPACION_TRAMO"] / kpis["ocupacion_promedio"]["CAPACIDAD_ASIENTOS_TRAMO"]

    ##################################################################
    # --- Tasa de venta anticipada
    kpis["venta_anticipada"] = (
        df.groupby( "DIAS_ANTICIPACION" ).agg({
            "BOLETOS_VEND": "sum",
            "VENTA": "sum",
            "TARIFA_BASE_TRAMO": "median"
        }).reset_index()
    )
    kpis["venta_anticipada"]['COSTO_PROM_BOLETO'] = kpis["venta_anticipada"]['VENTA'] / kpis["venta_anticipada"]['BOLETOS_VEND']

    ##################################################################
    # --- Top asientos más vendidos por hora y día
    df["NUM_ASIENTO"] = df["NUM_ASIENTO"].astype(int)
    kpis["venta_x_asientos"] = (
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
    kpis["venta_x_asientos"]["PAX_X_DIA_PROM"] = kpis["venta_x_asientos"]["PAX_SUBEN"] / kpis["venta_x_asientos"]["DIAS_VENTA"]

    # Calcular total de boletos por cada capacidad_asientos_tramo
    kpis["venta_x_asientos"] = kpis["venta_x_asientos"].sort_values(by=[ "CAPACIDAD_ASIENTOS_TRAMO", "PAX_X_DIA_PROM", "NUM_ASIENTO" ], ascending=[False, False, True] )
    # Paso 1: calcular el total de PAX_X_DIA_PROM por cada capacidad
    kpis["venta_x_asientos"]["TOTAL_PAX_X_DIA"] = kpis["venta_x_asientos"].groupby("CAPACIDAD_ASIENTOS_TRAMO")["PAX_X_DIA_PROM"].transform("sum")
    # Paso 2: calcular la proporción por asiento
    kpis["venta_x_asientos"]["PROP_ASIENTO"] = kpis["venta_x_asientos"]["PAX_X_DIA_PROM"] / kpis["venta_x_asientos"]["TOTAL_PAX_X_DIA"]
    # Paso 3: ordenar y calcular acumulado dentro de cada capacidad
    kpis["venta_x_asientos"]["PROP_ACUM"] = kpis["venta_x_asientos"].groupby("CAPACIDAD_ASIENTOS_TRAMO")["PROP_ASIENTO"].cumsum()
    # OPCIONAL: convertir a porcentaje (0–100 en lugar de 0–1)
    kpis["venta_x_asientos"]["PROP_ASIENTO_PCT"] = kpis["venta_x_asientos"]["PROP_ASIENTO"] * 100
    kpis["venta_x_asientos"]["PROP_ACUM_PCT"] = kpis["venta_x_asientos"]["PROP_ACUM"] * 100

    kpis["venta_x_asientos"] = kpis["venta_x_asientos"].reset_index(drop=True)


    ##################################################################
    # --- Diferencia de ventas por origen
    kpis["ventas_por_origen"] = (
        df.groupby("ORIGEN")[["BOLETOS_VEND", "VENTA"]].mean().reset_index()
    )

    ##################################################################
    # --- Picos de temporada (mes)
    kpis["ventas_por_mes"] = (
        df2.groupby(["AÑO", "MES"])[["BOLETOS_VEND", "VENTA", "PAX_SUBEN"]].sum().reset_index()
    )

    ##################################################################
    ## -- Punto de quiebre tarifario
    df_prueba = df.loc[df['VENTA_TOTAL']>0, :]
    df_prueba = df_prueba.loc[df_prueba['DIAS_ANTICIPACION']>=0, :]
    df_prueba = df_prueba.loc[df_prueba['FECHA_OPERACION'].dt.year>=2025, :]

    min_val = 250
    max_val = 1800
    bin_size = 25

    # Creamos los rangos desde 0 hasta >= max
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Creamos etiquetas para los grupos (ej: 0-50, 50-100, ...)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]

    # Segmentamos la columna
    df_prueba["RANGO_TARIFA"] = pd.cut(df_prueba["VENTA_TOTAL"], bins=bins, labels=labels, right=False, include_lowest=True)

    # Ahora agrupamos por rango de tarifa
    kpis["boletos_x_tarifa"] = df_prueba.groupby("RANGO_TARIFA").agg(
        VENTA_TOTAL = ("VENTA_TOTAL", "sum"),
        BOLETOS_VEND = ("BOLETOS_VEND", "sum"),
        PAX_SUBEN = ("PAX_SUBEN", "sum"),
        VECES_OFERTADA = ("OCUPACION_TRAMO", "size" ),
        DIAS_ANTICIPACION = ("DIAS_ANTICIPACION", "mean"),
        DIAS_UNICOS=("FECHA_OPERACION", "nunique")
    ).reset_index()

    kpis["boletos_x_tarifa"]['BOLETOS_PROM'] = kpis["boletos_x_tarifa"]['BOLETOS_VEND'] / kpis["boletos_x_tarifa"]['DIAS_UNICOS']
    kpis["boletos_x_tarifa"]['PAX_PROM'] = kpis["boletos_x_tarifa"]['PAX_SUBEN'] / kpis["boletos_x_tarifa"]['DIAS_UNICOS']

    return kpis