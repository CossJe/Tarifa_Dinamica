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
from functools import reduce
# Se extraen los datos del modelo de datos
#D4NN, D4C, D4C1, D4GC = T4DE.Get_Data()

D4NN = T4DE.Get_Data4NN()

#T4E.MainElas()
#TCD.BuenasCaracteristicas(D4GC)
#kpi= TCD.calcular_kpis(D4GC)

"""
Capacidad= kpi['CAPACIDAD_ASIENTOS_TRAMO'].unique()
Data={}
for cap in Capacidad:
    # Definimos el valor de la capacidad que quieres filtrar
    capacidad_objetivo = cap
    
    # Aplicamos los dos filtros usando el operador lógico '&' (AND)
    filtro = (kpi['CAPACIDAD_ASIENTOS_TRAMO'] == capacidad_objetivo) & (kpi['PROP_ACUM'] <= 0.60)
    
    # Aplicamos el filtro al DataFrame
    resultado = kpi[filtro]['NUM_ASIENTO']
    Data[cap]= list(resultado)
    
"""

"""
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

"""