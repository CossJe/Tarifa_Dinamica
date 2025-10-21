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

# Se extraen los datos del modelo de datos
D4NN, D4C, D4C1 = T4DE.Get_Data()

# se extrae la base de datos de los clusters generados
DBClus= T4DE.GetDB()
DataElas= T4E.GetDataElasticity()
Elas=DataElas['Elasticidad']

# Se obtienen los datos de los clientes para obtener la tarifa personalizada
TodayData4C= T4TP.GetTodayData4Cluster(D4C1)
TodayData4C['VENTA']= TodayData4C['TARIFA'].copy()
 TodayData4C= TodayData4C.drop('TARIFA',axis=1)
# 1. Inicializa una lista vacía para guardar los resultados
lista_resultados = []

# 2. Itera y agrega cada DataFrame a la lista
for row in range(len(TodayData4C)):
    # 1. Crea la fila (DataFrame de 1xN)
    InfoClient = pd.DataFrame(TodayData4C.iloc[row]).T
    
    # 2. Procesa la información (asumo que T4TP.GetCluster devuelve Cluster y desc)
    Cluster, desc = T4TP.GetCluster(InfoClient, DBClus,Elas)
    
    # 3. Agrega la nueva columna
    InfoClient["%_Tarifa Personalizada"] = desc
    InfoClient["Cluster"] = Cluster
    
    # 4. Agrega el DataFrame resultante a la lista
    lista_resultados.append(InfoClient)

# 3. Concatena todos los DataFrames de la lista de una sola vez
TodayDataTP = pd.concat(lista_resultados, ignore_index=True)