# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 13:28:22 2025

@author: Jesus Coss
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from Tools import Tools4DataExtraction as T4DE
from Tools.Tools4Cluster import ClusteringData
from Tools.Tools4ClasSupervisada import ClusteringSupervisado
from Tools import Tools4TarifPer as T4TP
from Tools import Tools4Net as T4N
from Tools import Tools4Elasticity as T4E
from Tools import Tools4CaracDemanda as TCD
import client_segmentation as CS
import kpis_calculator as KpiC
from functools import reduce
# Se extraen los datos del modelo de datos
D4NN, D4C, D4C1, D4GC = T4DE.Get_Data()

#T4E.MainElas()

"""
#TCD.BuenasCaracteristicas(D4GC)
kpi= KpiC.calcular_kpis(D4GC,'AÑO',2025)

VxA= kpi['venta_x_asientos']

Capacidad= VxA['CAPACIDAD_ASIENTOS_TRAMO'].unique()

frames=[]
for cap in Capacidad:
    # Definimos el valor de la capacidad que quieres filtrar
    capacidad_objetivo = cap
    
    # Aplicamos los dos filtros usando el operador lógico '&' (AND)
    filtro = (VxA['CAPACIDAD_ASIENTOS_TRAMO'] == capacidad_objetivo) & (VxA['PROP_ACUM'] <= 0.60)
    
    # Aplicamos el filtro al DataFrame
    resultado = VxA[filtro]['NUM_ASIENTO']
    frames.append(resultado)
    
# 1. Convertir cada Serie (lista de asientos) a un conjunto para la intersección
conjuntos_de_asientos = [set(frame) for frame in frames]

# 2. Usar reduce para encontrar la intersección común a todos los conjuntos
#    La intersección te da solo los elementos que están presentes en TODOS.
asientos_comunes_set = reduce(set.intersection, conjuntos_de_asientos)

# 3. (Opcional) Convertir el resultado de nuevo a una lista si lo necesitas
asientos_comunes_lista = list(asientos_comunes_set)
"""

""" """
# 0 es para todos los dias antes de ayer
# -1 es para todos los dias antes de 6 dias desde ayer
# 1 es desde ayer hasta hace un año
T4N.ProcessingNet(D4NN,-1)

# para pronosticar con todo el dataframe

# False es para el dia de hoy
# True es para hace 6 dias hasta hoy
df_= T4N.PrepareData4Fore(D4NN,True)

#Fore= T4N.GetTodayData4Net(df_)
#PrecioDin=T4N.NewClientsPredNet(Fore)
#df_['PRECIO DINAMICO']=PrecioDin
#TodayDataPD= df_.copy()

# para pronosticar por cliente el dataframe
lista_resultados = []

for row in range(len(df_)):
    InfoClient = pd.DataFrame(df_.iloc[row]).T
    Foree= T4N.GetTodayData4Net(InfoClient)
    InfoClient["TARIFA DINAMICA"] = T4N.NewClientsPredNet(Foree)[0,0]
    
    lista_resultados.append(InfoClient)

TodayDataPD = pd.concat(lista_resultados, ignore_index=True)

