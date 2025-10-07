# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:06:34 2025

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

from src.dynamic_pricing_data_loader import cargar_y_preparar_datos

def GetData():
    """
    Carga y normaliza el DataFrame con las columnas relevantes para el análisis de ventas/pasajeros.

    Flujo general
    ------------
    1. Determina las rutas (directorio actual y config.json).
    2. Llama a `cargar_y_preparar_datos(config_path, ruta_principal)` para obtener el DataFrame base.
    3. Selecciona un subconjunto de columnas relevantes.
    4. Filtra filas con 'VENTA_TOTAL' > 0.
    5. Rellena valores faltantes en 'PORCENT_PROMO' con 0.
    6. Completa la columna 'EMAIL' usando:
       - emails "conocidos" (mapeo por NOMBRE_PASAJERO),
       - o generando un email genérico con dominio 'ejemplo.com' para los restantes.
    7. Devuelve el DataFrame procesado (Df).

    Devuelve
    --------
    pandas.DataFrame
        DataFrame con las columnas:
        ['NOMBRE_PASAJERO','BOLETOS_VEND','CLASE_SERVICIO','DESC_DESCUENTO',
         'DIAS_ANTICIPACION','EMAIL','FECHA_CORRIDA','FECHA_OPERACION',
         'HORAS_ANTICIPACION','PAGO_METODO','PORCENT_PROMO','TIPO_CORRIDA',
         'TIPO_PASAJERO','TOTAL_BOLETOS','VENTA','VENTA_ANTICIPADA','VENTA_TOTAL']

    Notas y recomendaciones
    -----------------------
    - Requiere que exista la función externa `cargar_y_preparar_datos`.
    - Para evitar `SettingWithCopyWarning` es preferible usar asignaciones con `.loc`.
    - 'dominio' está hardcodeado como 'ejemplo.com'; considera parametrizarlo.
    """

    # Obtener el directorio de trabajo actual (ruta principal del proyecto).
    ruta_principal = os.getcwd()

    # Construir la ruta al archivo de configuración "config/config.json".
    config_path = os.path.join(ruta_principal, "config", "config.json")

    # Llamar a la función externa que carga y realiza preprocesamiento inicial.
    Frame = cargar_y_preparar_datos(config_path, ruta_principal)

    # Seleccionar solo las columnas relevantes para el análisis.
    Df = Frame[['NOMBRE_PASAJERO','BOLETOS_VEND', 'CLASE_SERVICIO', 'DESC_DESCUENTO', 'DIAS_ANTICIPACION',
                'EMAIL', 'FECHA_CORRIDA', 'FECHA_OPERACION', 'HORAS_ANTICIPACION',
                'PAGO_METODO', 'PORCENT_PROMO',
                'TIPO_CORRIDA', 'TIPO_PASAJERO',  'TOTAL_BOLETOS',
                'VENTA', 'VENTA_ANTICIPADA', 'VENTA_TOTAL']]

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

    # Devolver el DataFrame final procesado.
    return Df

def CompleteData4Cluster():
    """
    Procesa y completa información agregada a nivel de EMAIL para clustering.

    Flujo general
    -------------
    1. Obtiene el DataFrame base con `GetData()`.
    2. Genera variables agregadas por cliente (EMAIL):
       - Total de boletos vendidos (Bol_Vend).
       - Promedio pagado (Prom_Pagado).
       - % de promoción promedio (%_Promo).
       - Promedio de horas de anticipación (Prom_Horas_Ant).
    3. Calcula proporciones de tipos de descuento (DESC_DESCUENTO) y agrupa
       categorías en DMS (discapacidad/menor/senectud) y ACADEMICOS (estudiante/profesor).
    4. Calcula proporciones de métodos de pago (PAGO_METODO).
    5. Calcula el promedio de días entre compras por cliente (Prom_Dias_Entre_Compra),
       e identifica si fue la primera compra (Primer_Compra).
    6. Devuelve un DataFrame enriquecido (df_correo) listo para clustering.

    Devuelve
    --------
    pandas.DataFrame
        DataFrame con las variables agregadas y enriquecidas a nivel EMAIL.
    """

    # Obtener los datos procesados de la función GetData
    Df = GetData()    # Convertir columna de fecha a tipo datetime
    Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
    fecha_maxima = Df['FECHA_OPERACION'].max()
    Df = Df[Df['FECHA_OPERACION'] < fecha_maxima].copy()
    Df.loc[(Df['PORCENT_PROMO'] > 0) & (Df['DESC_DESCUENTO'] == 'ADULTO'), 'DESC_DESCUENTO'] = 'PROMOCION ESPECIAL'
    Df.loc[(Df['PORCENT_PROMO'] == 0) & (Df['DESC_DESCUENTO'] == 'PROMOCION ESPECIAL'), 'DESC_DESCUENTO'] = 'ADULTO'
    # Crear un DataFrame vacío para consolidar información por correo
    df_correo = pd.DataFrame()

    # Variable que servirá como clave de agrupación
    atributo = 'EMAIL'

    # Agregar variables básicas por cliente
    df_correo['Bol_Vend'] = Df.groupby(atributo)['BOLETOS_VEND'].sum()
    df_correo['Prom_Pagado'] = Df.groupby(atributo)['VENTA'].mean()
    df_correo['%_Promo'] = Df.groupby(atributo)['PORCENT_PROMO'].mean()
    df_correo['Prom_Horas_Ant'] = Df.groupby(atributo)['HORAS_ANTICIPACION'].mean()

    # Tabla de proporciones por tipo de descuento
    DESC_DESCUENTO = pd.crosstab(
        index=Df[atributo],
        columns=Df['DESC_DESCUENTO'],
        normalize='index'
    )

    # Reiniciar índice para que EMAIL sea columna y no índice
    DESC_DESCUENTO = DESC_DESCUENTO.reset_index()

    # Unir proporciones de descuento con el DataFrame principal
    df_correo = pd.merge(
        df_correo,
        DESC_DESCUENTO,
        on='EMAIL',
        how='left'
    )

    # --- Variables de métodos de pago ---

    # Crear tabla de proporciones por método de pago
    PAGO_METODO = pd.crosstab(
        index=Df[atributo],
        columns=Df['PAGO_METODO'],
        normalize='index'
    )

    # Reiniciar índice para que EMAIL sea columna
    PAGO_METODO = PAGO_METODO.reset_index()

    # Asegurar tipo de dato string en la columna clave
    PAGO_METODO[atributo] = PAGO_METODO[atributo].astype(str)

    # Unir la información de métodos de pago al DataFrame principal
    df_correo = pd.merge(
        df_correo,
        PAGO_METODO,
        on=atributo,
        how='left'
    )

    # Eliminar columna "EFECTIVO" ya que no se desea en el análisis
    df_correo = df_correo.drop(columns='EFECTIVO')

    # --- Variables temporales: días entre compras ---

    # Ordenar por cliente y fecha
    Df_sorted = Df.sort_values(['EMAIL', 'FECHA_OPERACION'])

    # Calcular diferencia en días entre compras consecutivas
    Df_sorted['DIF_DIAS'] = Df_sorted.groupby('EMAIL')['FECHA_OPERACION'].diff().dt.days

    # Promediar las diferencias de días por cliente
    PP = (
        Df_sorted
        .groupby('EMAIL')['DIF_DIAS']
        .mean()
        .reset_index(name='Prom_Dias_Entre_Compra')
    )

    # Crear variable binaria: 1 si es primera compra (sin historial), 0 en otro caso
    df_correo['Primer_Compra'] = PP['Prom_Dias_Entre_Compra'].isna()
    df_correo['Primer_Compra'] = df_correo['Primer_Compra'].astype(int)

    # Reemplazar NaN por 0 en promedios de días
    PP = PP.fillna(0)

    # Unir los promedios de días al DataFrame principal
    df_correo = df_correo.merge(
        PP,
        on='EMAIL',
        how='left'
    ).fillna({'Prom_Dias_Entre_Compra': 0})

    df_correo.to_excel('PorCorreo.xlsx')
    Df.to_excel('ventas.xlsx')
    # Devolver el DataFrame completo
    return df_correo

def CompleteData4Cluster1():
    """
    Construye un DataFrame consolidado por cliente (agrupado por correo) con 
    métricas relevantes para clustering.

    Flujo de la función:
    1. Obtiene los datos preprocesados de la función `GetData`.
    2. Reasigna la categoría "PROMOCION ESPECIAL" cuando aplica según el porcentaje de promoción.
    3. Calcula variables agregadas por correo (suma de boletos, monto promedio, total pagado, % promo, etc.).
    4. Calcula la proporción de compras con venta anticipada.
    5. Calcula la recencia (días desde la última compra hasta la fecha más reciente del dataset).
    6. Extrae la moda de métodos de pago y tipo de descuento por correo.
    7. Convierte las modas en variables dummy para usarse en clustering.
    8. Exporta resultados a Excel y devuelve el DataFrame final consolidado.

    Returns
    -------
    pd.DataFrame
        DataFrame con las métricas agregadas por correo para usar en clustering.
    """

    # 1. Obtener los datos procesados desde una función externa
    Df = GetData()
    Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
    fecha_maxima = Df['FECHA_OPERACION'].max()
    Df = Df[Df['FECHA_OPERACION'] < fecha_maxima].copy()
    # 2. Corrección de categorías:
    # Si un adulto tiene promoción > 0, renombrar su descuento como "PROMOCION ESPECIAL"
    Df.loc[(Df['PORCENT_PROMO'] > 0) & (Df['DESC_DESCUENTO'] == 'ADULTO'), 'DESC_DESCUENTO'] = 'PROMOCION ESPECIAL'
    
    # Si tiene 0% de promoción pero estaba marcado como "PROMOCION ESPECIAL", devolverlo a "ADULTO"
    Df.loc[(Df['PORCENT_PROMO'] == 0) & (Df['DESC_DESCUENTO'] == 'PROMOCION ESPECIAL'), 'DESC_DESCUENTO'] = 'ADULTO'
    
    # 3. Crear un DataFrame vacío donde consolidaremos info a nivel correo
    df_correo = pd.DataFrame()

    # Variable clave de agrupación (identificador del cliente)
    atributo = 'EMAIL'
    
    # 4. Calcular métricas agregadas por cliente (EMAIL)
    df_correo['SBol_Vend'] = Df.groupby(atributo)['BOLETOS_VEND'].sum()      # Total de boletos vendidos
    # df_correo['PBol_Vend'] = Df.groupby(atributo)['BOLETOS_VEND'].mean()   # Promedio de boletos vendidos (comentado)
    df_correo['Prom_Pagado'] = Df.groupby(atributo)['VENTA'].mean()          # Ticket promedio de compra
    df_correo['Sum_Pagado'] = Df.groupby(atributo)['VENTA'].sum()            # Monto total gastado
    df_correo['%_Promo'] = Df.groupby(atributo)['PORCENT_PROMO'].mean()      # Promedio de % de promoción usado
    df_correo['Prom_Horas_Ant'] = Df.groupby(atributo)['HORAS_ANTICIPACION'].mean()  # Anticipación promedio de compra

    # 5. Proporción de compras con venta anticipada usando crosstab (frecuencias relativas)
    prop_ct = pd.crosstab(
        index=Df[atributo],
        columns=Df['VENTA_ANTICIPADA'],
        normalize='index'   # normaliza por cliente → proporción
    )
    
    # Agregar al DataFrame la proporción de compras con "SI" en venta anticipada
    df_correo['Venta_Ant'] = prop_ct['SI'].fillna(0)
    
    # 6. Calcular "Recencia" (días desde última compra de cada cliente)
    
    # Fecha máxima de compra por cliente
    df_max = Df.groupby('EMAIL')['FECHA_OPERACION'].max().reset_index(name='FECHA_MAX')
    
    # Fecha máxima global (última compra registrada en el dataset)
    fecha_max_global = Df['FECHA_OPERACION'].max()
    
    # Diferencia en días entre la última compra global y la última compra del cliente
    df_max['Recencia'] = (fecha_max_global - df_max['FECHA_MAX']).dt.days
    
    # Añadir la columna de recencia al DataFrame consolidado
    df_correo = df_correo.merge(df_max[['EMAIL', 'Recencia']], on='EMAIL', how='left')
    
    # 7. Obtener la moda de variables categóricas por cliente (pago y tipo de descuento)
    df_modas = Df.groupby('EMAIL').agg({
        'PAGO_METODO': lambda x: x.mode()[0],       # método de pago más frecuente
        'DESC_DESCUENTO': lambda x: x.mode()[0]     # tipo de descuento más frecuente
    }).reset_index()
    
    # Renombrar columnas de salida
    df_modas = df_modas.rename(columns={
        'PAGO_METODO': 'Tipo_pago',
        'DESC_DESCUENTO': 'Tipo_desc'
    })
    
    # 8. Crear variables dummies para métodos de pago y tipos de descuento
    df_dummies_pago = pd.get_dummies(df_modas['Tipo_pago'], prefix='PAGO').astype(int)
    df_dummies_desc = pd.get_dummies(df_modas['Tipo_desc'], prefix='DESC').astype(int)
    
    # Concatenar dummies al DataFrame consolidado
    df_correo = pd.concat([df_correo, df_dummies_pago, df_dummies_desc], axis=1)
    
    # 9. Exportar resultados a Excel
    df_correo.to_excel('PorCorreo.xlsx')  # DataFrame agregado por cliente
    Df.to_excel('ventas.xlsx')            # Dataset original con correcciones
    
    # 10. Devolver el DataFrame consolidado para análisis/clustering
    return df_correo

def GetClustersMoreThanOne(df,optimal_k=0):
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

    # Guarda el modelo en el archivo
    joblib.dump(modelo_entrenado, nombre_archivo)
    
    print(f"Modelo K-Means guardado exitosamente como: {nombre_archivo}")
    
    # 9. Asignar a cada cliente su cluster
    dfMoreThanOne['Cluster'] = modelo_entrenado.labels_
    
    # 10. Devolver el DataFrame filtrado y con clusters
    return dfMoreThanOne

def GetClusters4One(df, n):
    """
    Filtra a los clientes con exactamente una compra y les aplica clustering (KMeans),
    escalando las variables con RobustScaler para reducir la influencia de outliers.
    Finalmente, ajusta la numeración de los clusters sumando una constante `n`.

    Flujo de la función:
    1. Filtra clientes con `SBol_Vend == 1`.
    2. Selecciona solo las variables numéricas (excluyendo EMAIL y anteriores).
    3. Escala las variables con `RobustScaler`.
    4. Evalúa diferentes valores de K (2 a 9) usando el coeficiente de Silueta 
       para encontrar el número óptimo de clusters.
    5. Entrena un modelo KMeans con el K óptimo.
    6. Asigna los clusters al DataFrame filtrado.
    7. Suma `n` al número de cada cluster (para mantener consistencia si se combina con otros grupos).
    8. Devuelve el DataFrame final con los clusters asignados.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame consolidado (ej. salida de CompleteData4Cluster1), 
        donde cada fila representa un cliente (EMAIL).
    
    n : int
        Número entero que se suma a los valores de los clusters para 
        desplazar su numeración (ej. evitar solapamientos con otros grupos).

    Returns
    -------
    pd.DataFrame
        Subconjunto de clientes con una sola compra y columna 'Cluster' 
        que indica su grupo, ajustado con el desplazamiento `n`.
    """

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

    # Guarda el modelo en el archivo
    joblib.dump(modelo_entrenado, nombre_archivo)
    
    print(f"Modelo K-Means guardado exitosamente como: {nombre_archivo}")
    # 9. Asignar clusters al DataFrame filtrado
    dfOne['Cluster'] = modelo_entrenado.labels_
    
    # 10. Ajustar numeración sumando la constante 'n'
    # (para evitar solapamientos con otros conjuntos de clusters)
    dfOne = dfOne.copy()
    dfOne['Cluster'] = dfOne['Cluster'] + n

    # 11. Devolver DataFrame con clusters ajustados
    return dfOne


