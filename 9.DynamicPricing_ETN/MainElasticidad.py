# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:19:14 2025

@author: Jesus Coss
"""
import json

import Tools4Elasticidad as Ts

def MainElas():
    """
    Función principal para calcular la elasticidad de la demanda y sugerir precios.

    Esta función orquesta un flujo de trabajo que incluye la carga de datos,
    la ingeniería de características, el cálculo de la elasticidad de la demanda
    y la sugerencia de precios óptimos.

    Parámetros:
    -----------
    UltimaTar : float, opcional
        La última tarifa conocida, utilizada como un valor de referencia para
        los cálculos de precios. Por defecto es 1163.79.

    Retorna:
    --------
    tuple
        - PrecioMaximo (float): El precio máximo sugerido para la venta.
        - PrecioSugerido (float): El precio sugerido para optimizar los ingresos.
        - Elasticidad (float): El valor de la elasticidad precio de la demanda.
    """
    # Se definen las condiciones iniciales para el pronóstico.
    CondIni = {
        'Dias_Anticipacion': 0,
        'Mes_Viaje': 12,
        'Fin_Semana_Viaje': 0,
        'Buen_dia': 1
    }
    
    # Se obtienen y preparan los datos para el año 2024.
    Frame = Ts.GetData(2024)
    TBT= Frame['TARIFA'].iloc[-1]
    Frame = Ts.GetFeatures(Frame)
    
    # Se calculan los coeficientes del modelo de elasticidad y el valor de la elasticidad.
    Coef, Elasticidad = Ts.GetElasticity(Frame)
    
    # Se utilizan los coeficientes y las condiciones iniciales para sugerir precios.
    PrecioMaximo, PrecioSugerido = Ts.GetPrizes(Coef, CondIni, TBT)
    
    # Convertir a tipos nativos de Python
    data = {
        "TBT": int(TBT),
        "PrecioMaximo": float(PrecioMaximo),
        "PrecioSugerido": float(PrecioSugerido),
        "Elasticidad": float(Elasticidad)
    }
    
    with open("Resultados_Elasticidad.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return 

MainElas()