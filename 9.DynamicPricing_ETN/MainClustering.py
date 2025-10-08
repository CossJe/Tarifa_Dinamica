# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:53:08 2025

@author: Jesus Coss
"""
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import Tools4Cluster as TC



def MainCluster(bandera):
    df=TC.CompleteData4Cluster1()
    if bandera:
        df1=TC.GetClustersMoreThanOne(df,6)
        n=max(df1['Cluster'])+1
        df2=TC.GetClusters4One(df,n)
        
        df_final = pd.concat([df1, df2], ignore_index=True)
    else:
        df_final =  TC.GetCluster4AllData(df,6)
        
    cluster_profile = df_final.groupby('Cluster')[df_final.columns[1:]].mean().round(2)
    with pd.ExcelWriter('ClusteringClientes.xlsx') as writer:
        df_final.to_excel(writer, sheet_name='Clustering', index=False)
        cluster_profile.to_excel(writer, sheet_name='Resumen', index=False)
    

MainCluster(False)
