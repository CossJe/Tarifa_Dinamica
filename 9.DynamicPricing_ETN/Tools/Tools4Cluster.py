# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:22:12 2025

@author: Jesus Coss
"""

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import json
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import joblib

def ModifyingData(Frame):
    Df=Frame.copy()
    
    Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
    fecha_maxima = Df['FECHA_OPERACION'].max()
    Df = Df[Df['FECHA_OPERACION'] < fecha_maxima].copy()
    
    # Filtrar registros con VENTA_TOTAL > 0 (elimina ventas nulas o negativas).
    Df=Df[Df[ 'VENTA_TOTAL']>0]
    Df=Df[Df[ 'BOLETOS_VEND']>0]
    Df['HORAS_ANTICIPACION']=Df['HORAS_ANTICIPACION'].abs()
    Df['DIAS_ANTICIPACION']=Df['DIAS_ANTICIPACION'].abs()

    # Rellenar valores faltantes en PORCENT_PROMO con 0.
    Df['PORCENT_PROMO'] = Df['PORCENT_PROMO'].fillna(0)

    # Construir un DataFrame 'known' con nombres y emails conocidos:
    # - eliminar filas sin EMAIL,
    # - eliminar duplicados por NOMBRE_PASAJERO,
    # - quedarnos solo con ['NOMBRE_PASAJERO', 'EMAIL'].
    known = (
        Df
        .dropna(subset=['EMAIL'])
        .drop_duplicates(subset=['NOMBRE_PASAJERO'])
        [['NOMBRE_PASAJERO','EMAIL']]
    )

    # Hacer un merge para anotar cada fila con el EMAIL conocido (si existe).
    # Se crea temporalmente la columna 'EMAIL_KNOWN' para contener el email mapeado.
    Df = Df.merge(
        known,
        on='NOMBRE_PASAJERO',
        how='left',
        suffixes=('','_KNOWN')
    )

    # Rellenar la columna EMAIL con EMAIL_KNOWN cuando EMAIL original esté vacío.
    # Esto prioriza el email original; si es NaN, toma el conocido.
    Df['EMAIL'] = Df['EMAIL'].fillna(Df['EMAIL_KNOWN'])

    # Eliminar la columna auxiliar EMAIL_KNOWN que ya no se necesita.
    Df.drop(columns=['EMAIL_KNOWN'], inplace=True)

    # Dominio que se usará para generar emails genéricos (placeholder).
    dominio = 'ejemplo.com'

    # Obtener la lista de nombres únicos sin email asignado.
    nombres_sin_email = Df.loc[Df['EMAIL'].isna(), 'NOMBRE_PASAJERO'].unique()

    # Crear un diccionario mapping nombre -> email genérico (ej: "juan.perez@ejemplo.com").
    # Nota: se hace una transformación simple (lower + reemplazar espacios por puntos).
    generic_map = {
        nombre: f"{nombre.lower().replace(' ','.')}@{dominio}"
        for nombre in nombres_sin_email
    }

    # Asegurar que la columna EMAIL es de tipo object (cadena).
    Df['EMAIL'] = Df['EMAIL'].astype('object')

    # Rellenar los emails faltantes usando el mapping generado.
    # Se usa map sobre NOMBRE_PASAJERO y fillna para no sobrescribir emails existentes.
    Df['EMAIL'] = Df['EMAIL'].fillna(Df['NOMBRE_PASAJERO'].map(generic_map))
    
    # Corrección de categorías:
    # Si un adulto tiene promoción > 0, renombrar su descuento como "PROMOCION ESPECIAL"
    Df.loc[(Df['PORCENT_PROMO'] > 0) & (Df['DESC_DESCUENTO'] == 'ADULTO'), 'DESC_DESCUENTO'] = 'PROMOCION ESPECIAL'
    
    # Si tiene 0% de promoción pero estaba marcado como "PROMOCION ESPECIAL", devolverlo a "ADULTO"
    Df.loc[(Df['PORCENT_PROMO'] == 0) & (Df['DESC_DESCUENTO'] == 'PROMOCION ESPECIAL'), 'DESC_DESCUENTO'] = 'ADULTO'
    # Devolver el DataFrame final procesado.
    return Df


def CompleteData4Cluster(Frame,ruta_principal):
    # 1. Obtener los datos procesados desde una función externa
    Df = ModifyingData(Frame)
    
    # 2. Crear un DataFrame vacío donde consolidaremos info a nivel correo
    df_correo = pd.DataFrame()

    # Variable clave de agrupación (identificador del cliente)
    atributo = 'EMAIL'
    
    # 3. Calcular métricas agregadas por cliente (EMAIL)
    df_correo['SBol_Vend'] = Df.groupby(atributo)['BOLETOS_VEND'].sum()      # Total de boletos vendidos
    # df_correo['PBol_Vend'] = Df.groupby(atributo)['BOLETOS_VEND'].mean()   # Promedio de boletos vendidos (comentado)
    df_correo['Prom_Pagado'] = Df.groupby(atributo)['VENTA'].mean()          # Ticket promedio de compra
    df_correo['Sum_Pagado'] = Df.groupby(atributo)['VENTA'].sum()            # Monto total gastado
    df_correo['%_Promo'] = Df.groupby(atributo)['PORCENT_PROMO'].mean()      # Promedio de % de promoción usado
    df_correo['Prom_Horas_Ant'] = Df.groupby(atributo)['HORAS_ANTICIPACION'].mean()  # Anticipación promedio de compra

    # 4. Proporción de compras con venta anticipada usando crosstab (frecuencias relativas)
    prop_ct = pd.crosstab(
        index=Df[atributo],
        columns=Df['VENTA_ANTICIPADA'],
        normalize='index'   # normaliza por cliente → proporción
    )
    
    # Agregar al DataFrame la proporción de compras con "SI" en venta anticipada
    df_correo['Venta_Ant'] = prop_ct['SI'].fillna(0)
    
    # 5. Calcular "Recencia" (días desde última compra de cada cliente)
    
    # Fecha máxima de compra por cliente
    df_max = Df.groupby('EMAIL')['FECHA_OPERACION'].max().reset_index(name='FECHA_MAX')
    
    # Fecha máxima global (última compra registrada en el dataset)
    fecha_max_global = Df['FECHA_OPERACION'].max()
    
    # Diferencia en días entre la última compra global y la última compra del cliente
    df_max['Recencia'] = (fecha_max_global - df_max['FECHA_MAX']).dt.days
    
    # Añadir la columna de recencia al DataFrame consolidado
    df_correo = df_correo.merge(df_max[['EMAIL', 'Recencia']], on='EMAIL', how='left')
    
    # 6. Obtener la moda de variables categóricas por cliente (pago y tipo de descuento)
    df_modas = Df.groupby('EMAIL').agg({
        'PAGO_METODO': lambda x: x.mode()[0],       # método de pago más frecuente
        'DESC_DESCUENTO': lambda x: x.mode()[0]     # tipo de descuento más frecuente
    }).reset_index()
    
    # Renombrar columnas de salida
    df_modas = df_modas.rename(columns={
        'PAGO_METODO': 'Tipo_pago',
        'DESC_DESCUENTO': 'Tipo_desc'
    })
    
    # 7. Crear variables dummies para métodos de pago y tipos de descuento
    df_dummies_pago = pd.get_dummies(df_modas['Tipo_pago'], prefix='PAGO').astype(int)
    df_dummies_desc = pd.get_dummies(df_modas['Tipo_desc'], prefix='DESC').astype(int)
    
    # Concatenar dummies al DataFrame consolidado
    df_correo = pd.concat([df_correo, df_dummies_pago, df_dummies_desc], axis=1)
    
    # 8. Exportar resultados a Excel
    config_path = os.path.join(ruta_principal, "Files", "PorCorreo.csv")
    df_correo.to_csv(config_path, index=False)  # DataFrame agregado por cliente
    config_path = os.path.join(ruta_principal, "Files", "ventas.csv")
    Df.to_csv(config_path, index=False)          # Dataset original con correcciones
    
    #9. Devolver el DataFrame consolidado para análisis/clustering
    return df_correo

def GetClustersMoreThanOne(df,ruta_principal, optimal_k=0):
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
    #optimal_k = 0
    
    # Rango de valores de K a evaluar
    K_range = range(2, 10)
    
    if optimal_k == 0: 
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

    nombre_archivo = "Kmeans4MoreThanOne.joblib"
    config_path = os.path.join(ruta_principal, "Models", nombre_archivo)
    # Guarda el modelo en el archivo
    joblib.dump(modelo_entrenado, config_path)
    
    print(f"Modelo K-Means guardado exitosamente como: {nombre_archivo}")
    
    # 9. Asignar a cada cliente su cluster
    dfMoreThanOne['Cluster'] = modelo_entrenado.labels_
    
    # 10. Devolver el DataFrame filtrado y con clusters
    return dfMoreThanOne

def GetClusters4One(df, n,ruta_principal):
    # 1. Filtrar clientes con exactamente 1 boleto vendido
    dfOne = df[df['SBol_Vend'] == 1].copy()
    
    # 2. Ubicar la posición de la columna EMAIL
    Col = list(df.columns).index('EMAIL')
    
    # 3. Seleccionar variables numéricas (excluyendo EMAIL y previas)
    X = dfOne[dfOne.columns[Col+1:]]
    
    # --- Escalado ---
    # 4. Crear instancia de escalador robusto (menos sensible a outliers)
    robust_scaler = RobustScaler()
    
    # 5. Ajustar y transformar las variables numéricas
    X_escalado = robust_scaler.fit_transform(X)
    
    # 6. Convertir el resultado a DataFrame con nombres originales
    X_escalado = pd.DataFrame(X_escalado, columns=X.columns)
    # --- Fin del paso de escalado ---
    
    # --- Selección de K óptima ---
    max_silhouette_score = -1   # valor inicial del coeficiente de silueta
    optimal_k = 0               # número óptimo de clusters
    K_range = range(2, 10)      # valores de K a evaluar
    
    # 7. Probar diferentes valores de K y seleccionar el mejor
    for k in K_range:
        kmeans_model = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans_model.fit(X_escalado)
        
        # Calcular score de silueta
        score = silhouette_score(X_escalado, kmeans_model.labels_)
        
        # Guardar el mejor resultado
        if score > max_silhouette_score:
            max_silhouette_score = score
            optimal_k = k
    
    # Mostrar el número óptimo de clusters
    print(f"El número óptimo de clusters (K) es: {optimal_k}")
    
    # --- Entrenamiento del modelo final ---
    # 8. Entrenar KMeans con la K óptima encontrada
    modelo_entrenado = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    modelo_entrenado.fit(X_escalado)
    
    nombre_archivo = "Kmeans4One.joblib"
    config_path = os.path.join(ruta_principal, "Models", nombre_archivo)

    # Guarda el modelo en el archivo
    joblib.dump(modelo_entrenado, config_path)
    
    print(f"Modelo K-Means guardado exitosamente como: {nombre_archivo}")
    # 9. Asignar clusters al DataFrame filtrado
    dfOne['Cluster'] = modelo_entrenado.labels_
    
    # 10. Ajustar numeración sumando la constante 'n'
    # (para evitar solapamientos con otros conjuntos de clusters)
    dfOne = dfOne.copy()
    dfOne['Cluster'] = dfOne['Cluster'] + n

    # 11. Devolver DataFrame con clusters ajustados
    return dfOne


def GetCluster4AllData(df,optimal_k,ruta_principal):
    # 1. Ubicar la posición de la columna EMAIL (para separar variables de identificación)
    Col = list(df.columns).index('EMAIL')
    
    #2. Tomar solo las variables numéricas (excluyendo EMAIL y columnas previas a ella)
    X = df[df.columns[Col+1:]]
    
    # --- Escalado ---
    # 3. Crear instancia del escalador Min-Max
    minmax_scaler = MinMaxScaler()
    
    # 4. Ajustar y transformar las variables numéricas
    X_escalado = minmax_scaler.fit_transform(X)
    
    # 5. Convertir el resultado a DataFrame con mismos nombres de columnas
    X_escalado = pd.DataFrame(X_escalado, columns=X.columns)
    # --- Fin del escalado ---    
    
    # --- Entrenamiento del modelo final ---
    # 8. Entrenar KMeans con la K óptima encontrada
    modelo_entrenado = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    modelo_entrenado.fit(X_escalado)
    
    nombre_archivo = "KmeansAllData.joblib"
    config_path = os.path.join(ruta_principal, "Models", nombre_archivo)
    
    # Guarda el modelo en el archivo
    joblib.dump(modelo_entrenado, config_path)
    
    print(f"Modelo K-Means guardado exitosamente como: {nombre_archivo}")
    # 9. Asignar clusters al DataFrame filtrado
    df['Cluster'] = modelo_entrenado.labels_
    
    return df

def ClusteringData(bandera,Frame):
    ruta_principal = os.getcwd()
    df=CompleteData4Cluster(Frame,ruta_principal)
    if bandera:
        df1=GetClustersMoreThanOne(df,6)
        n=max(df1['Cluster'])+1
        df2=GetClusters4One(df,n)
        
        df_final = pd.concat([df1, df2], ignore_index=True)
    else:
        df_final =  GetCluster4AllData(df,6,ruta_principal)
    
    cluster_profile = df_final.groupby('Cluster')[df_final.columns[1:]].mean().round(2)
    # 1. Guarda el DataFrame df_final (el que era la hoja 'Clustering')
    csv_path_clustering = os.path.join(ruta_principal, "Files", "ClusteringClientes_Clustering.csv")
    df_final.to_csv(csv_path_clustering, index=False)
    
    # 2. Guarda el DataFrame cluster_profile (el que era la hoja 'Resumen')
    csv_path_resumen = os.path.join(ruta_principal, "Files", "ClusteringClientes_Resumen.csv")
    cluster_profile.to_csv(csv_path_resumen, index=False)