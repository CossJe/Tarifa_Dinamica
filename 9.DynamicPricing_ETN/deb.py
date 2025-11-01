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
    D4NN = Frame[["PAX_SUBEN", 'FECHA_OPERACION','VENTA','ORIGEN', 'DESTINO','BOLETOS_VEND']].copy()
    D4NN= D4NN[(D4NN['ORIGEN']=='MEXN') & (D4NN['DESTINO']== 'GDLJ')].copy()
    D4NN=D4NN.iloc[:-1]
    return D4NN




D4NN = Get_Data()

df_agg = D4NN.groupby(['FECHA_OPERACION']).agg(
    # Definición correcta de 3 nuevas columnas promediando cada original
    Promedio_Venta=('VENTA', 'mean'),
    Promedio_Pax=('PAX_SUBEN', 'mean'),
    Q_Pax=('PAX_SUBEN', 'count'),
    # Esta parte se ve bien
    Q_Boletos=('BOLETOS_VEND', 'count')
).reset_index()


# 1. Encontrar la fecha máxima
fecha_maxima = df_agg['FECHA_OPERACION'].max()
# 2. Calcular el día anterior a la fecha máxima
dia_anterior = fecha_maxima - timedelta(days=1)
# 3. Calcular la fecha de inicio (365 días antes del día_anterior)
fecha_inicio = dia_anterior - timedelta(days=364) # Para incluir 365 días, es decir, día_anterior y los 364 previos.
# 4. Filtrar el DataFrame
# Se incluyen todas las fechas en el rango [fecha_inicio, dia_anterior]
Df_365 = df_agg[(df_agg['FECHA_OPERACION'] >= fecha_inicio) & 
            (df_agg['FECHA_OPERACION'] <= dia_anterior)].copy()


plt.plot(Df_365['FECHA_OPERACION'], Df_365['Q_Boletos'])
plt.show()

Df_365['lnQ'] = np.log(df_agg['Q_Boletos'])
Df_365['lnP'] = np.log(df_agg['Promedio_Venta'])

model = smf.ols('lnQ ~ lnP', data=Df_365).fit()
coeficientes = model.params
b1 = model.params['lnP']
print(f"El valor de la elasticidad es {b1}")


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


#T4E.MainElas()

demanda_predicha_base = 1
TBT= 1163
Elas= b1
resultado_optimizacion = encontrar_precio_optimo(
    demanda_base=demanda_predicha_base,
    tarifa_base=TBT,
    elasticidad=Elas # El valor que obtuvimos en la Fase 2
)

precio_final = resultado_optimizacion['precio_optimo']
ingreso_final = resultado_optimizacion['ingreso_maximo']


# --- PASO E: MOSTRAR RESULTADOS Y GRÁFICO ---
print(f"\nPara un viaje con una demanda base predicha de {demanda_predicha_base:.0f} boletos (a ${TBT:.2f}):")
print("\n--------------------------------------------------")
print(f"  El precio óptimo recomendado es: ${precio_final:.2f}")
print(f"  Con este precio, el ingreso esperado se maximiza a: ${ingreso_final:,.2f}")
print("--------------------------------------------------")

# Graficar
df_simulacion = resultado_optimizacion['datos_simulacion']
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
sns.lineplot(data=df_simulacion, x='precio_probado', y='ingreso_esperado', linewidth=3)
plt.axvline(x=precio_final, color='red', linestyle='--', label=f'Precio Óptimo (${precio_final:.2f})')
plt.axhline(y=ingreso_final, color='red', linestyle='--')
plt.title('Curva de Optimización de Ingresos', fontsize=18)
plt.xlabel('Precio del Boleto ($)', fontsize=12)
plt.ylabel('Ingreso Total Esperado ($)', fontsize=12)
plt.legend()
plt.show()