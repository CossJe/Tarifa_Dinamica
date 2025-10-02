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

def GetClustersMoreThanOne(df):
    """
    Filtra los clientes con más de una compra y les aplica clustering (KMeans) 
    para segmentarlos en grupos de comportamiento similares.

    Flujo de la función:
    1. Identifica clientes con más de 1 boleto vendido.
    2. Escala las variables numéricas (Min-Max Scaling).
    3. Busca el número óptimo de clusters (K) usando el coeficiente de Silueta.
    4. Entrena un modelo KMeans con ese K óptimo.
    5. Devuelve el DataFrame filtrado con la asignación de cluster.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame consolidado (ej. salida de CompleteData4Cluster1), 
        donde cada fila representa un cliente (EMAIL).

    Returns
    -------
    pd.DataFrame
        Subconjunto del DataFrame con clientes con más de una compra y 
        una nueva columna 'Cluster' que indica a qué grupo pertenece cada cliente.
    """

    # 1. Ubicar la posición de la columna EMAIL (para separar variables de identificación)
    Col = list(df.columns).index('EMAIL')
    
    # 2. Filtrar clientes con más de 1 boleto vendido
    dfMoreThanOne = df[df['SBol_Vend'] > 1].copy()
    
    # 3. Tomar solo las variables numéricas (excluyendo EMAIL y columnas previas a ella)
    X = dfMoreThanOne[dfMoreThanOne.columns[Col+1:]]
    
    # --- Escalado ---
    # 4. Crear instancia del escalador Min-Max
    minmax_scaler = MinMaxScaler()
    
    # 5. Ajustar y transformar las variables numéricas
    X_escalado = minmax_scaler.fit_transform(X)
    
    # 6. Convertir el resultado a DataFrame con mismos nombres de columnas
    X_escalado = pd.DataFrame(X_escalado, columns=X.columns)
    
    # --- Fin del escalado ---
    
    # --- Selección de K óptima ---
    # Inicializar variables para guardar el mejor valor de silueta y K
    max_silhouette_score = -1
    optimal_k = 0
    
    # Rango de valores de K a evaluar
    K_range = range(2, 10)
    
    # 7. Probar diferentes valores de K y elegir el que dé mejor coeficiente de silueta
    for k in K_range:
        kmeans_model = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans_model.fit(X_escalado)
        
        # Calcular score de silueta
        score = silhouette_score(X_escalado, kmeans_model.labels_)
        
        # Actualizar K óptimo si mejora
        if score > max_silhouette_score:
            max_silhouette_score = score
            optimal_k = k
    
    # Mostrar el número óptimo de clusters
    print(f"El número óptimo de clusters (K) es: {optimal_k}")
    
    # --- Entrenamiento del modelo final ---
    # 8. Entrenar modelo KMeans con la K óptima encontrada
    modelo_entrenado = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    modelo_entrenado.fit(X_escalado)
    
    # 9. Asignar a cada cliente su cluster
    dfMoreThanOne['Cluster'] = modelo_entrenado.labels_
    
    # 10. Devolver el DataFrame filtrado y con clusters
    return dfMoreThanOne

def GetClusters4One(df):
    """
    Filtra a los clientes con exactamente una compra y les aplica clustering (KMeans) 
    para segmentarlos en grupos. Usa un escalado robusto para reducir el impacto de outliers.

    Flujo de la función:
    1. Selecciona clientes con una sola compra (`SBol_Vend == 1`).
    2. Escala las variables numéricas con `RobustScaler`.
    3. Evalúa diferentes valores de K (2 a 9) y selecciona el óptimo con el coeficiente de Silueta.
    4. Entrena un modelo KMeans con el K óptimo.
    5. Asigna los clusters al DataFrame filtrado.
    6. Reasigna los clusters 0 → 7 y 1 → 8 (probablemente para unificar criterios con otros segmentos).
    7. Devuelve el DataFrame con clientes de una sola compra y su cluster asignado.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame consolidado (ej. salida de CompleteData4Cluster1), 
        donde cada fila representa un cliente (EMAIL).

    Returns
    -------
    pd.DataFrame
        Subconjunto de clientes con una sola compra y columna 'Cluster' 
        que indica su grupo.
    """

    # 1. Filtrar clientes con exactamente 1 boleto vendido
    dfOne = df[df['SBol_Vend'] == 1].copy()
    
    # 2. Ubicar la posición de la columna EMAIL
    Col = list(df.columns).index('EMAIL')
    
    # 3. Seleccionar variables numéricas (excluyendo EMAIL y previas)
    X = dfOne[dfOne.columns[Col+1:]]
    
    # --- Escalado ---
    # 4. Crear instancia de escalador robusto (menos sensible a outliers que Min-Max o Standard)
    robust_scaler = RobustScaler()
    
    # 5. Ajustar y transformar variables numéricas
    X_escalado = robust_scaler.fit_transform(X)
    
    # 6. Convertir el resultado en DataFrame con nombres originales
    X_escalado = pd.DataFrame(X_escalado, columns=X.columns)
    # --- Fin del paso de escalado ---
    
    # --- Selección de K óptima ---
    max_silhouette_score = -1   # valor inicial
    optimal_k = 0               # mejor número de clusters encontrado
    K_range = range(2, 10)      # rango de clusters a probar
    
    # 7. Probar diferentes valores de K y seleccionar el mejor según silueta
    for k in K_range:
        kmeans_model = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans_model.fit(X_escalado)
        
        # Calcular score de silueta
        score = silhouette_score(X_escalado, kmeans_model.labels_)
        
        # Actualizar el óptimo si mejora
        if score > max_silhouette_score:
            max_silhouette_score = score
            optimal_k = k
    
    # Mostrar el número óptimo de clusters
    print(f"El número óptimo de clusters (K) es: {optimal_k}")
    
    # --- Entrenamiento del modelo final ---
    # 8. Entrenar modelo KMeans con la K óptima
    modelo_entrenado = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    modelo_entrenado.fit(X_escalado)
    
    # 9. Asignar cluster a cada cliente
    dfOne['Cluster'] = modelo_entrenado.labels_
    
    # 10. Reasignar etiquetas de cluster:
    #     - Cluster 0 pasa a ser 7
    #     - Cluster 1 pasa a ser 8
    # (Esto se hace probablemente para diferenciarlos de los clusters de clientes con más de una compra)
    dfOne['Cluster'] = np.where(dfOne['Cluster'] == 0, 7, dfOne['Cluster'])
    dfOne['Cluster'] = np.where(dfOne['Cluster'] == 1, 8, dfOne['Cluster'])

    # 11. Devolver el DataFrame final
    return dfOne

def MainCluster():
    df=TC.CompleteData4Cluster1()
    df1=GetClustersMoreThanOne(df)
    df2=GetClusters4One(df)
    
    df_final = pd.concat([df1, df2], ignore_index=True)
    cluster_profile = df_final.groupby('Cluster')[df_final.columns[1:]].mean().round(2)
    with pd.ExcelWriter('ClusteringClientes.xlsx') as writer:
        df_final.to_excel(writer, sheet_name='Clustering', index=False)
        cluster_profile.to_excel(writer, sheet_name='Resumen', index=False)
    
    
MainCluster()
