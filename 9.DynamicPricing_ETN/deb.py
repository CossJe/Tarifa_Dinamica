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

T4N.ProcessingNet(D4NN)

df_= T4N.PrepareData4Fore(D4NN)
#Fore= T4N.GetTodayData4Net(df_)
#PrecioDin=T4N.NewClientsPredNet(Fore)

lista_resultados = []

for row in range(len(df_)):
    InfoClient = pd.DataFrame(df_.iloc[row]).T
    Foree= T4N.GetTodayData4Net(InfoClient)
    InfoClient["PRECIO DINAMICO"] = T4N.NewClientsPredNet(Foree)[0,0]
    
    lista_resultados.append(InfoClient)

TodayDataPD = pd.concat(lista_resultados, ignore_index=True)