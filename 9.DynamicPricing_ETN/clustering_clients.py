from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def preparar_datos(df, features):
    """
    Prepara los datos para clustering:
    - Selección de features
    - One-Hot Encoding para variables categóricas
    - Eliminación de outliers en numéricas (2.5% - 97.5%)
    
    Parámetros:
        df (pd.DataFrame): DataFrame original
        features (list): Lista de columnas a usar
    
    Retorna:
        pd.DataFrame: DataFrame procesado y listo para clustering
    """
    
    # Seleccionamos solo las columnas requeridas
    data = df[features].copy()
    
    # Separamos categóricas y numéricas
    cat_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()
    num_vars = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # --- 1. One-Hot Encoding para categóricas ---
    if cat_vars:
        data = pd.get_dummies(data, columns=cat_vars, drop_first=True)
    
    # --- 2. Tratamiento de outliers en numéricas (Winsorization) ---
    for col in num_vars:
        # winsorize devuelve un array → lo volvemos serie
        data[col] = winsorize(data[col], limits=[0.025, 0.025])
    
    # Resetear índice tras eliminación
    data = data.reset_index(drop=True)
    
    return data



def run_kmeans(X_scaled, n_clusters=5, random_state=42):
    """
    Ejecuta KMeans sobre los datos escalados.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans



def run_dbscan(X_scaled, eps=1.5, min_samples=10):
    """
    Ejecuta DBSCAN sobre los datos escalados.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    return labels, dbscan



def segment_clients(df, method="kmeans", features=None, **kwargs):
    """
    Segmenta clientes/transacciones con KMeans o DBSCAN.

    Parámetros:
    -----------
    df : DataFrame con tus datos
    method : str, "kmeans" o "dbscan"
    features : list, columnas numéricas a usar
    kwargs : parámetros específicos para cada método

    Retorna:
    --------
    df : DataFrame con columna 'cluster'
    model : objeto del modelo entrenado
    """
    if features is None:
        # Variables numéricas recomendadas
        features = [
            "DIAS_ANTICIPACION",
            "VENTA_TOTAL",
            "BOLETOS_VEND",
            "KMS_TRAMO",
            "PORCENT_PROMO",
            "OCUPACION_TRAMO"
        ]
    
    # 1. Selección y limpieza
    X = df[features].fillna(0)

    # 2. Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Ejecutar clustering
    if method == "kmeans":
        labels, model = run_kmeans(X_scaled, **kwargs)
    elif method == "dbscan":
        labels, model = run_dbscan(X_scaled, **kwargs)
    else:
        raise ValueError("Método no reconocido. Usa 'kmeans' o 'dbscan'.")

    # 4. Guardar resultados
    df = df.copy()
    df["cluster"] = labels

    return df, model



def cluster_summary(df, features, cluster_col="cluster"):
    """
    Genera un resumen con la media de las variables por cluster.
    """
    resumen = df.groupby(cluster_col)[features].mean().round(2)
    conteos = df[cluster_col].value_counts().rename("count")
    return resumen.join(conteos, how="left")



def plot_clusters(df, features, cluster_col="cluster"):
    """
    Visualiza clusters en 2D usando PCA.
    """
    X = df[features].fillna(0)
    
    # Reducción a 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], 
                          c=df[cluster_col], cmap="tab10", alpha=0.7)