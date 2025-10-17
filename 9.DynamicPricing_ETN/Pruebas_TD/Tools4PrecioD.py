# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:15:38 2025

@author: Jesus Coss
"""
import pandas as pd
import numpy as np


def encontrar_precio_optimo(demanda_base, tarifa_base, elasticidad, rango_precios=(0.7, 1.5), pasos=100):
    """
    Simula diferentes precios para encontrar el que maximiza el ingreso total.

    Args:
        demanda_base (float): La demanda predicha por el modelo de Fase 1 a la tarifa base.
        tarifa_base (float): El precio de referencia histórico para el viaje.
        elasticidad (float): El coeficiente de elasticidad precio de la demanda.
        rango_precios (tuple): Rango de precios a probar como multiplicador de la tarifa_base.
        pasos (int): Número de precios a probar en el rango.

    Returns:
        dict: Un diccionario con el precio óptimo, el ingreso máximo y datos para graficar.
    """
    # 1. Crear un rango de precios para probar
    precios_a_probar = np.linspace(
        tarifa_base * rango_precios[0],  # Precio mínimo
        tarifa_base * rango_precios[1],  # Precio máximo
        pasos
    )

    ingresos_esperados = []
    demandas_simuladas = []

    # 2. Simular la demanda y el ingreso para cada precio
    for nuevo_precio in precios_a_probar:
        # Calcular el cambio porcentual en el precio
        cambio_pct_precio = (nuevo_precio - tarifa_base) / tarifa_base

        # Usar la elasticidad para calcular el cambio porcentual en la demanda
        cambio_pct_demanda = elasticidad * cambio_pct_precio

        # Calcular la nueva demanda simulada
        nueva_demanda = demanda_base * (1 + cambio_pct_demanda)

        # Asegurarse de que la demanda no sea negativa
        if nueva_demanda < 0:
            nueva_demanda = 0

        demandas_simuladas.append(nueva_demanda)

        # Calcular el ingreso esperado para este precio
        ingreso = nuevo_precio * nueva_demanda
        ingresos_esperados.append(ingreso)

    # 3. Encontrar el precio que maximizó el ingreso
    ingreso_maximo = max(ingresos_esperados)
    indice_optimo = ingresos_esperados.index(ingreso_maximo)
    precio_optimo = precios_a_probar[indice_optimo]

    return {
        "precio_optimo": precio_optimo,
        "ingreso_maximo": ingreso_maximo,
        "datos_simulacion": pd.DataFrame({
            'precio_probado': precios_a_probar,
            'demanda_simulada': demandas_simuladas,
            'ingreso_esperado': ingresos_esperados
        })
    }

