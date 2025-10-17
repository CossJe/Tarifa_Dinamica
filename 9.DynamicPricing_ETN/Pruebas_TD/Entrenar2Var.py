# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:24:16 2025

@author: Jesus Coss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

####  -------------------------------------------------------------------------------------  
#           Preparacion data 
####  -------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

def Get_goodDay(df, atribute):
    """
    Identifica y marca los 3 días de la semana con mayor volumen de ventas.

    Esta función procesa un DataFrame con un índice de tipo fecha para
    determinar los tres días de la semana (por ejemplo, lunes, martes,
    miércoles) que tienen la suma más alta del atributo de ventas especificado.
    Luego, crea una nueva columna llamada 'Es buen dia', que se establece en 1
    si la fecha en esa fila cae en uno de esos tres días de la semana de alto
    rendimiento, y en 0 en caso contrario.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame de entrada con un índice de tipo fecha y la columna de
        ventas a analizar.
    atribute : str
        El nombre de la columna en `df` que contiene los valores de ventas
        (por ejemplo, 'VENTA_BOLETOS').

    Retorna:
    --------
    tuple
        Una tupla que contiene:
        - df_output : pandas.DataFrame
            Un nuevo DataFrame que es una copia del original con dos columnas
            adicionales:
            - 'Dia de la semana': El número del día de la semana (0=lunes,
              6=domingo).
            - 'Es buen dia': Un valor entero (1 si el día de la semana es uno
              de los 3 mejores, 0 en caso contrario).
        - dias : list
            Una lista de enteros que representa los índices de los 3 días de
            la semana con las ventas más altas (por ejemplo, [5, 6, 4] para
            sábado, domingo y viernes).
    """

    df_output = df.copy() # Se crea una copia para evitar modificar el DataFrame original.
    
    # Se agrega una columna con el nombre del día de la semana para el análisis.
    df_output['Dia de la semana'] = df_output.index.day_name()
    
    # Se agrupan los datos por día de la semana y se suma el atributo
    # especificado para obtener las ventas diarias totales.
    daily_sales = df_output.groupby('Dia de la semana')[atribute].sum()
    
    # Se ordenan las ventas de forma descendente para encontrar los mejores días.
    daily_sales = daily_sales.sort_values(ascending=False)
    
    # Se seleccionan los 3 días con las ventas más altas.
    daily_sales = daily_sales[:3]
    
    # Se crea un diccionario para mapear los nombres de los días a sus
    # respectivos índices numéricos (0-6).
    indices = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    # Se obtienen los índices numéricos de los 3 mejores días de la semana.
    dias = list(daily_sales.index.map(indices).to_numpy())
    
    # Se crea la columna 'Es buen dia' usando el índice del día de la semana
    # y verificando si está en la lista de los 3 mejores días. Se convierte a
    # tipo entero (1 o 0).
    df_output['Es buen dia'] = df_output.index.dayofweek.isin(dias).astype(int)
    
    # Se asegura de que la columna 'Es buen dia' tenga el tipo de dato correcto.
    df_output['Es buen dia'] = df_output['Es buen dia'].astype(np.int64)
    
    # Finalmente, se reemplaza el nombre del día de la semana por su índice numérico
    # en la columna 'Dia de la semana' del DataFrame de salida.
    df_output['Dia de la semana'] = df_output.index.dayofweek
    df_output['Dia de la semana'] = df_output['Dia de la semana'].astype(np.int64)
    
    # Se retorna el DataFrame modificado y la lista de los mejores días.
    return df_output, dias


def agregar_lags_por_dia_semana(df):
    """
    Agrega variables de retardo (lags) a un DataFrame por día de la semana.

    Esta función calcula los valores de retardo (lag) para una columna de
    destino, pero en lugar de hacerlo de manera lineal (un día a la vez),
    lo hace por día de la semana. Esto significa que un valor de un lunes
    se desplaza para compararse con el lunes de la semana anterior (lag 7),
    con el lunes de hace dos semanas (lag 14), y así sucesivamente. Esto es
    particularmente útil para datos con estacionalidad semanal.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame de entrada que debe contener la columna a la que se le
        calcularán los lags y una columna llamada 'Dia de la semana' con
        valores de 0 a 6 (donde 0 es lunes).

    Retorna:
    --------
    pandas.DataFrame
        Una copia del DataFrame original con tres nuevas columnas agregadas:
        - '{nombre_columna}_lag_7': El valor de la semana anterior.
        - '{nombre_columna}_lag_14': El valor de hace dos semanas.
        - '{nombre_columna}_lag_21': El valor de hace tres semanas.
    """

    # Se crea una copia del DataFrame para no modificar el original.
    df_output = df.copy()
    
    # Se identifica la columna objetivo, asumiendo que es la primera columna.
    target_col = df_output.columns[0]
    
    # Se inicializan las nuevas columnas de lag con valores NaN. Esto asegura
    # que existan en el DataFrame antes de llenarlos.
    for lag in [7, 14, 21]:
        df_output[f'{target_col}_lag_{lag}'] = np.nan
    
    # Se itera sobre cada día de la semana (0 a 6).
    for day in range(7):
        # Se filtra el DataFrame para obtener solo las filas que corresponden
        # a un día específico (ej. todos los lunes).
        subset_dia = df_output[df_output['Dia de la semana'] == day].copy()
        
        # Se calculan los lags para este subconjunto. El `shift(1)` en este
        # subconjunto equivale a desplazar 7 días en el DataFrame principal.
        subset_dia[f'{target_col}_lag_7'] = subset_dia[target_col].shift(1)
        subset_dia[f'{target_col}_lag_14'] = subset_dia[target_col].shift(2)
        subset_dia[f'{target_col}_lag_21'] = subset_dia[target_col].shift(3)
        
        # Se usa `update` para unir los lags calculados en el subconjunto de
        # vuelta al DataFrame principal. Esto es eficiente porque solo actualiza
        # las filas correspondientes.
        df_output.update(subset_dia)
    
    # Se retorna el DataFrame con las nuevas columnas de lags.
    return df_output

def agregar_caracteristicas_fecha(df):
    """
    Agrega características de la fecha a un DataFrame.

    Esta función enriquece un DataFrame de series de tiempo al extraer
    características temporales del índice de fecha. Esto es útil para
    modelos de aprendizaje automático, ya que convierte la información
    cíclica de la fecha en variables numéricas que pueden ser utilizadas
    como predictores.

    Args:
        df (pd.DataFrame): El DataFrame de entrada que debe tener un
                           índice de tipo fecha (datetime).

    Returns:
        pd.DataFrame: Una copia del DataFrame original con las siguientes
                      nuevas columnas de características de fecha:
                      - 'DAP': Un valor booleano (True/False) que indica
                        si la primera columna del DataFrame está por
                        encima del promedio de sus valores.
                      - 'dia_del_mes': El día del mes (1-31).
                      - 'dia_del_año': El día del año (1-365 o 366).
                      - 'semana_del_año': El número de la semana del año
                        según el calendario ISO 8601.
    """
    # Se crea una copia del DataFrame de entrada para no modificar el original.
    df_con_fecha = df.copy()

    # Se crea la característica 'DAP' (Día por encima del promedio).
    # Se asume que la primera columna del DataFrame es la variable objetivo
    # y se compara cada valor con la media de toda la columna.
    df_con_fecha['DAP'] = (df_con_fecha[df_con_fecha.columns[0]] >= df_con_fecha[df_con_fecha.columns[0]].mean())

    # Se extrae el día del mes del índice de fecha y se añade como una nueva columna.
    df_con_fecha['dia_del_mes'] = df_con_fecha.index.day

    # Se extrae el día del año del índice de fecha.
    df_con_fecha['dia_del_año'] = df_con_fecha.index.dayofyear

    # Las siguientes líneas están comentadas pero muestran la intención
    # de agregar más características como el mes y variables de interacción.
    # df_con_fecha['mes'] = df_con_fecha.index.month
    # df_con_fecha['semana_del_año'] = df_con_fecha.index.isocalendar().week.astype(int)

    # Se extrae el número de la semana del año (ISO) y se convierte a tipo entero.
    # Esto es útil para capturar patrones semanales o estacionalidad.
    df_con_fecha['semana_del_año'] = df_con_fecha.index.isocalendar().week.astype(int)

    # df_con_fecha['mes_x_dia_semana'] = df_con_fecha['mes'] * df_con_fecha['Dia de la semana']
    # df_con_fecha['mes_x_es_buen_dia'] = df_con_fecha['mes'] * df_con_fecha['Es buen dia']

    # Se retorna el DataFrame modificado con las nuevas columnas.
    return df_con_fecha

def Obtener_Lags(df):
    """
    Calcula y sugiere los lags más relevantes de una serie de tiempo
    basándose en el análisis de autocorrelación.

    Esta función utiliza la Autocorrelación (ACF) y la Autocorrelación Parcial
    (PACF) para encontrar los lags que son estadísticamente significativos.
    El resultado es una lista de lags que pueden ser utilizados como
    características en modelos de pronóstico, priorizando la estacionalidad
    semanal si es relevante.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame de entrada que contiene una serie de tiempo univariada.
        Se asume que la serie es la primera columna del DataFrame.

    Retorna:
    --------
    list
        Una lista de tres enteros que representan los lags sugeridos para ser
        utilizados en un modelo de pronóstico o análisis.
    """
    # Se crea una copia del DataFrame para evitar modificar el original.
    df_ = df.copy()
    n = 30  # Se define el número máximo de lags a considerar para el análisis.

    # Se calcula la Función de Autocorrelación (ACF) y la Función de
    # Autocorrelación Parcial (PACF).
    # - `acf`: Mide la correlación entre la serie y sus valores rezagados.
    # - `pacf`: Mide la correlación entre la serie y sus valores rezagados,
    #           eliminando el efecto de los lags intermedios.
    acf_valores, acf_conf = acf(df_, nlags=n, alpha=0.05)
    pacf_valores, pacf_conf = pacf(df_, nlags=n, alpha=0.05, method='ywm')

    # Encontrar los lags significativos. Se comparan los valores absolutos
    # de ACF y PACF con los límites de su banda de confianza (alpha=0.05).
    # Un valor fuera de esta banda es estadísticamente significativo.
    # El `[1:]` se usa para ignorar el lag 0, que siempre es 1.
    acf_significativos = np.where(np.abs(acf_valores[1:]) > (acf_conf[1:, 1] - acf_valores[1:]))[0] + 1
    pacf_significativos = np.where(np.abs(pacf_valores[1:]) > (pacf_conf[1:, 1] - pacf_valores[1:]))[0] + 1

    # La lógica condicional para seleccionar los lags a retornar.
    if 7 in acf_significativos:
        # Si el lag 7 (correspondiente a la estacionalidad semanal) es
        # significativo en la ACF, se prioriza este patrón.
        lags = [min(acf_significativos), 7, 14]
    else:
        # De lo contrario, se seleccionan los lags más importantes detectados
        # por la ACF. Se eligen el más pequeño, el más grande y un múltiplo
        # del más grande para capturar diferentes escalas de dependencia.
        lags = [min(acf_significativos), max(acf_significativos), max(acf_significativos) * 2]

    return lags


def agregar_lags(df, target_col):
    """
    Agrega lags y una media móvil de 7 días a un DataFrame para una columna objetivo.

    Esta función automatiza el proceso de ingeniería de características
    creando nuevas columnas con valores de series de tiempo rezagados (lags)
    y una media móvil para cada uno. Los lags son seleccionados dinámicamente
    por la función 'Obtener_Lags'. Esto es crucial para modelos de pronóstico
    porque la predicción del futuro se basa en el comportamiento pasado.

    Args:
        df (pd.DataFrame): DataFrame de entrada que contiene la serie de tiempo.
        target_col (str): Nombre de la columna objetivo a la que se le
                          aplicarán los lags y la media móvil.

    Returns:
        tuple: Una tupla que contiene:
               - pd.DataFrame: El DataFrame con las nuevas columnas
                 'lag_{n}' y 'Rolling_Mean_lag_{n}'.
               - list: La lista de los lags que se agregaron al DataFrame.
    """
    # Se crea una copia del DataFrame para no modificar el original.
    df_con_lags = df.copy()
    
    # Se obtienen los lags significativos de manera dinámica utilizando
    # la función 'Obtener_Lags'. Esto hace que el proceso sea más flexible.
    lags = Obtener_Lags(df_con_lags)

    # Se itera sobre la lista de lags obtenidos para crear las nuevas columnas.
    for lag in lags:
        # Se crea una nueva columna llamada 'lag_{lag}' que contiene
        # los valores de 'target_col' desplazados 'lag' posiciones.
        df_con_lags[f'lag_{lag}'] = df_con_lags[target_col].shift(lag)
        
        # Se calcula la media móvil de 2 días para el lag actual.
        # El .shift(lag) es crucial para evitar el "data leakage" (fuga de datos),
        # ya que la media móvil se calcula sobre valores pasados.
        df_con_lags[f'Rolling_Mean_lag_{lag}'] = (
            df_con_lags[target_col].shift(lag).rolling(window=2).mean())
    
    # La función retorna el DataFrame modificado y la lista de lags utilizados.
    return df_con_lags, lags

def agregar_componentes(df):
    """
    Descompone una serie de tiempo en sus componentes de tendencia,
    estacionalidad y residuos, y los agrega al DataFrame.

    Esta función utiliza la descomposición estacional para separar la serie
    en sus elementos clave. Esto permite analizar cada componente de forma
    independiente y puede mejorar los modelos de pronóstico, ya que estos
    componentes pueden ser predichos o modelados por separado.

    Args:
        df (pd.DataFrame): DataFrame de entrada que contiene una serie de
                           tiempo univariada, que se asume es la primera
                           columna. El índice debe ser de tipo fecha
                           para que la descomposición funcione correctamente.

    Returns:
        pd.DataFrame: Un DataFrame que es una copia del original con dos
                      nuevas columnas agregadas: 'tendencia' y
                      'estacionalidad'.
    """
    # Se crea una copia del DataFrame para no modificar el original.
    df_original = df.copy()

    # Realizar la descomposición de la serie.
    # El modelo 'additive' (aditivo) se utiliza cuando la magnitud de la
    # estacionalidad no cambia con la tendencia.
    # El 'period=7' indica que la estacionalidad es semanal.
    descomposicion = seasonal_decompose(df_original[df_original.columns[0]], model='additive', period=7)

    # Extraer los componentes de la descomposición.
    tendencia = descomposicion.trend
    estacionalidad = descomposicion.seasonal
    residuos = descomposicion.resid

    # Se unen los componentes al DataFrame principal.
    df_original['tendencia'] = tendencia
    df_original['estacionalidad'] = estacionalidad
    
    # La siguiente línea está comentada, pero muestra la intención de
    # también agregar los residuos al DataFrame. Los residuos representan
    # la parte de la serie que queda después de remover la tendencia y la
    # estacionalidad, y a menudo son la parte más difícil de predecir.
    # df_original['residuos'] = residuos
    
    # Se retorna el DataFrame con los nuevos componentes.
    return df_original

def Obtener_Frec_Estac(df):
    """
    Detecta automáticamente la frecuencia estacional dominante en una
    serie de tiempo utilizando la Transformada Rápida de Fourier (FFT).

    Esta función convierte la serie de tiempo del dominio del tiempo al
    dominio de la frecuencia para identificar los componentes cíclicos más
    fuertes. El período estacional principal se determina a partir de la
    frecuencia con la mayor magnitud de su componente de Fourier.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame de entrada que contiene una serie de tiempo univariada.
        Se asume que la serie de tiempo es la primera columna del DataFrame.

    Retorna:
    --------
    int
        Un entero que representa el período estacional dominante detectado
        en la serie de tiempo. Por ejemplo, si se detecta un patrón semanal,
        el valor devuelto será 7.
    """
    # Se crea una copia del DataFrame para no modificar el original.
    df_con_fourier = df.copy()

    # Se extrae la serie de tiempo como un array de NumPy para el análisis.
    senal = df_con_fourier[df_con_fourier.columns[0]].to_numpy()
    n = len(senal)

    # Se aplica la Transformada Rápida de Fourier (FFT) a la señal.
    # Esto descompone la serie de tiempo en sus componentes de frecuencia.
    fft_result = np.fft.fft(senal)
    # Se obtienen las frecuencias correspondientes a los resultados de la FFT.
    freq = np.fft.fftfreq(n)

    # Se encuentran las magnitudes de las frecuencias, excluyendo la componente
    # de frecuencia 0 (media, o componente DC) y las frecuencias negativas.
    magnitudes = np.abs(fft_result)[1:n // 2]
    
    # Se encuentra la frecuencia con la mayor magnitud. Esta es la frecuencia
    # "dominante" o la que tiene el patrón cíclico más fuerte.
    freq_dominante = freq[1:n // 2][np.argmax(magnitudes)]

    # El período estacional es el inverso de la frecuencia dominante.
    # Por ejemplo, si la frecuencia dominante es 1/7, el período es 7.
    periodo_estacional = int(1 / freq_dominante)

    # Se retorna el período estacional.
    return periodo_estacional


def agregar_fourier_terms(df, n, n_terms):
    """
    Agrega términos de Fourier para capturar la estacionalidad.

    Esta función crea pares de columnas de seno y coseno (términos de Fourier)
    para representar la estacionalidad cíclica de los datos. Esta técnica
    es una alternativa eficaz al one-hot encoding para modelar patrones
    estacionales, ya que puede capturar relaciones más complejas y es más
    eficiente para períodos largos.

    Args:
        df (pd.DataFrame): DataFrame con un índice de fecha.
        n_terms (int): El número de términos de Fourier a agregar. Cada
                       término consta de un par seno/coseno.
        n (int): El paso para generar los términos de Fourier.

    Returns:
        pd.DataFrame: DataFrame original con las nuevas columnas de
                      términos de Fourier.
    """
    # Se crea una copia del DataFrame para evitar modificar el original.
    df_con_fourier = df.copy()

    # Se llama a la función para obtener la frecuencia estacional.
    freq = Obtener_Frec_Estac(df_con_fourier)

    # Se sobrescribe la frecuencia detectada para forzar una estacionalidad
    # semanal (periodo de 7 días). Esto hace que la función sea rígida.
    freq = 7

    # El bucle genera los términos de Fourier (pares de seno y coseno).
    for i in range(1, n_terms + 1, n):
        # Se crean las columnas de seno y coseno. La fórmula convierte el
        # número del día de la semana en un valor cíclico entre 0 y 2*pi.
        df_con_fourier[f'sin_{freq}_{i}'] = np.sin(2 * np.pi * i * df_con_fourier.index.dayofweek / 7)
        df_con_fourier[f'cos_{freq}_{i}'] = np.cos(2 * np.pi * i * df_con_fourier.index.dayofweek / 7)

    # Se retorna el DataFrame modificado.
    return df_con_fourier

def detectar_outliers_con_zscore(df, column, umbral=2):
    """
    Detecta y marca valores atípicos (outliers) en una columna de un DataFrame
    utilizando el método Z-Score.

    El Z-Score mide cuántas desviaciones estándar se encuentra un punto de datos
    respecto a la media. Esta función calcula el Z-Score para cada valor en la
    columna especificada y marca los puntos que superan un umbral predefinido
    como outliers.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame de entrada que contiene los datos a analizar.
    column : str
        El nombre de la columna en la que se buscarán los outliers.
    umbral : float, opcional
        El valor umbral del Z-Score para considerar un punto como outlier.
        Por defecto es 2. Un umbral de 2 generalmente captura aproximadamente
        el 95% de los datos si la distribución es normal. Un umbral de 3
        sería más estricto (99.7%).

    Retorna:
    --------
    pandas.DataFrame
        Una copia del DataFrame original con una nueva columna llamada
        'es_outlier'. Esta columna contiene 1 si el valor es un outlier
        (supera el umbral) y 0 si no lo es.
    """
    # Se crea una copia del DataFrame para no modificar el original.
    df_copy = df.copy()

    # Calcular el valor absoluto del Z-Score para cada elemento de la columna.
    # El Z-Score de un punto se calcula como (valor - media) / desviación_estándar.
    # stats.zscore devuelve los Z-Scores para cada punto del array de entrada.
    df_copy['zscore'] = np.abs(stats.zscore(df_copy[column]))

    # Se crea la columna 'es_outlier'. Si el Z-Score absoluto de un valor
    # es mayor que el umbral, se le asigna 1, de lo contrario, 0.
    df_copy['es_outlier'] = (df_copy['zscore'] > umbral).astype(int)

    # Se elimina la columna auxiliar 'zscore' del DataFrame final.
    # 'inplace=True' hace que la eliminación se realice directamente
    # en el DataFrame y no retorne una nueva copia.
    df_copy.drop('zscore', axis=1, inplace=True)

    # Se retorna el DataFrame con la nueva columna de outliers.
    return df_copy

def agregar_predictoras(df, df1):
    """
    Combina dos DataFrames para crear un conjunto de datos para el
    entrenamiento de modelos de series de tiempo.

    Esta función toma un DataFrame que ya contiene las características
    (predictoras) y le añade la columna de la variable objetivo
    ('VENTA'), extraída de un segundo DataFrame. Esto es un paso común
    en el preprocesamiento de datos para series de tiempo, donde
    se preparan las variables de entrada y la variable de salida
    para un modelo de machine learning.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame que contiene las variables predictoras (características).
        Se espera que este DataFrame tenga un índice de tiempo coincidente
        con `df1`.
    df1 : pandas.DataFrame
        El DataFrame que contiene la variable objetivo, que se añadirá
        como una nueva columna en el DataFrame de salida. Se asume que
        la columna objetivo es la primera columna.

    Retorna:
    --------
    pandas.DataFrame
        Un nuevo DataFrame que combina las características de `df` y la
        variable objetivo de `df1` en una única estructura de datos.
    """
    # Se crea una copia del DataFrame de características para no modificar el original.
    df_features = df.copy()

    # Se añade la columna de la variable objetivo al DataFrame de características.
    # Se asume que la variable objetivo es la primera columna de df1.
    # El `.values` se usa para asegurar que los valores se copien sin
    # los índices, evitando problemas de alineación si los índices
    # no coinciden perfectamente, aunque idealmente deberían.
    df_features['VENTA'] = df1[df1.columns[0]].values

    # Se retorna el DataFrame final que contiene tanto las características
    # como la variable objetivo.
    return df_features
    
def actualizar_data(df):
    """
    Actualiza y limpia un DataFrame eliminando las filas con valores faltantes.

    Esta función es una medida de preprocesamiento de datos que garantiza
    que el DataFrame esté completo y listo para ser utilizado en modelos
    o análisis posteriores. La función trabaja con una copia del DataFrame
    original para evitar efectos secundarios no deseados.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame de entrada que se va a limpiar.

    Retorna:
    --------
    pandas.DataFrame
        Un nuevo DataFrame que es una copia del original pero con todas
        las filas que contienen al menos un valor faltante eliminadas.
    """
    # Se crea una copia del DataFrame de entrada para no modificar el original.
    df_final = df.copy()

    # Se obtiene la última fecha del índice, aunque esta variable no se utiliza
    # en la lógica actual de la función. Podría ser un remanente de una
    # funcionalidad anterior o una variable de marcador de posición.
    last_date = df_final.index.max()

    # Se eliminan todas las filas que contienen valores faltantes (NaN o NaT).
    # `inplace=True` modifica el DataFrame directamente sin retornar una nueva copia,
    # lo cual en este caso es redundante ya que se va a retornar `df_final`.
    df_final.dropna(inplace=True)

    # Se retorna el DataFrame final, limpio de valores faltantes.
    return df_final

def Obtener_Ing_De_Carac(df_original, atribute, Param, df1, c):
    """
    Orquesta un pipeline de ingeniería de características para series de tiempo.

    Esta función aplica una serie de transformaciones y creaciones de variables
    a un DataFrame original. El objetivo es enriquecer los datos de la serie
    de tiempo con información temporal y estadística que puede mejorar
    significativamente el rendimiento de los modelos de pronóstico.

    Parámetros:
    -----------
    df_original : pandas.DataFrame
        El DataFrame de entrada con la serie de tiempo original.
    atribute : str
        El nombre de la columna que contiene los valores de la serie de tiempo,
        utilizado en funciones internas como Get_goodDay.
    Param : list
        Una lista de parámetros para las funciones internas, por ejemplo:
        [umbral_zscore, n_paso_fourier, n_terminos_fourier].
    df1 : pandas.DataFrame
        El DataFrame que contiene la variable objetivo, que se añadirá al final
        del proceso.
    c : int
        Un indicador para determinar el tipo de retorno. Si c=1, se devuelve
        un diccionario con los parámetros clave utilizados en el proceso.

    Retorna:
    --------
    tuple or pandas.DataFrame
        Si c=1, retorna una tupla que contiene:
        - El DataFrame final con todas las características.
        - Un diccionario con los parámetros clave utilizados.
        Si c es cualquier otro valor, retorna solo el DataFrame final.
    """
    # Se crea una copia del DataFrame original para no modificarlo.
    df_features = df_original.copy()

    # Se agrega una columna con los lags de la serie de tiempo.
    df_features, lags = agregar_lags(df_features, df_features.columns[0])

    # Se agregan los días de la semana con mayor volumen de ventas.
    df_features, dias = Get_goodDay(df_features, atribute)

    # Se agregan lags específicos para cada día de la semana.
    df_features = agregar_lags_por_dia_semana(df_features)

    # Se extraen características de la fecha como el día del mes o del año.
    df_features = agregar_caracteristicas_fecha(df_features)

    # La siguiente línea está comentada, lo que indica que la descomposición de
    # componentes no está activada en este momento.
    # df_features = agregar_componentes(df_features)

    # Se agregan los términos de Fourier para capturar la estacionalidad.
    df_features = agregar_fourier_terms(df_features, Param[1], Param[2])

    # Se detectan y marcan los outliers en la columna objetivo.
    df_features = detectar_outliers_con_zscore(df_features, df_features.columns[0], Param[0])

    # La siguiente línea está comentada, lo que indica que la eliminación
    # de filas con NaN no está activada en este momento.
    # df_features = actualizar_data(df_features)

    # Se añade la variable objetivo al DataFrame de características.
    df_features = agregar_predictoras(df_features, df1)

    # Lógica de retorno condicional.
    if c == 1:
        # Se crea un diccionario con un resumen de las características clave.
        carac = {'lags': lags, 'freq': 7, 'n_terms': Param[2], 'n': Param[1],
                 'rate': Param[0], 'dias': dias}
        return df_features, carac

    # Retorna solo el DataFrame de características.
    return df_features

####  -------------------------------------------------------------------------------------  
#           Buscador de parametros  
####  -------------------------------------------------------------------------------------

def GridSearch(df, Emp, name, df1):
    """
    Realiza una búsqueda de hiperparámetros para optimizar un modelo
    de pronóstico de series de tiempo.

    Esta función itera sobre un conjunto predefinido de hiperparámetros
    para encontrar la combinación que produce el mejor rendimiento (medido
    por el coeficiente R²) en el conjunto de prueba. Utiliza un pipeline de
    ingeniería de características y entrenamiento de modelos para cada
    iteración.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame de la serie de tiempo original.
    Emp : str
        El nombre de la entidad o empresa, usado para la impresión y gráficos.
    name : str
        El nombre de la columna objetivo.
    df1 : pandas.DataFrame
        El DataFrame que contiene la variable objetivo, que se añadirá
        al final del proceso.

    Retorna:
    --------
    tuple: Una tupla que contiene:
        - carac (dict): Un diccionario con los parámetros óptimos encontrados
          durante la búsqueda.
        - modelo_xgb (objeto): El modelo XGBoost final entrenado con los
          parámetros óptimos.
    """
    # Se inicializa el R² óptimo en 0.0.
    r2 = 0.0
    
    # Se definen los valores de los hiperparámetros a probar.
    Enes = [1, 1, 2, 4, 6]
    Arm = [2, 5, 10, 20, 30]

    # Se inicializan las variables que guardarán los mejores parámetros.
    Param = None
    df_aux = None # Se inicializa para evitar un error en caso de que los bucles no se ejecuten.

    # Bucle anidado para probar todas las combinaciones de hiperparámetros.
    for rate in np.arange(1.2, 3, 0.1):
        for n, n_terms in zip(Enes, Arm):
            # Se crea una copia del DataFrame para cada iteración.
            df_aux = df.copy()

            # Se aplica el pipeline de ingeniería de características con los parámetros actuales.
            df_features = Obtener_Ing_De_Carac(df_aux, name, (rate, n, n_terms), df1, 0)

            # Se entrena y evalúa el modelo, obteniendo el R² como resultado.
            r = EntrenamientoGS(df, df_features, Emp, name)

            # Se compara el R² actual con el mejor R² encontrado hasta el momento.
            if r > r2:
                # Si es mejor, se actualiza el R² óptimo y se guardan los parámetros.
                r2 = r
                Param = (rate, n, n_terms)

    # Una vez que se encuentra la mejor combinación, se vuelve a aplicar
    # el pipeline de ingeniería de características con los parámetros óptimos.
    # El parámetro `c=1` indica que se debe retornar el DataFrame final y los
    # parámetros de las características.
    df_features, carac = Obtener_Ing_De_Carac(df_aux, name, Param, df1, 1)

    # Se realiza un entrenamiento y visualización final con los mejores parámetros.
    modelo_xgb = Entrenamiento(df, df_features, Emp, name)

    # Se retornan los parámetros óptimos y el modelo entrenado.
    return carac, modelo_xgb

def EntrenamientoGS(df, df_features, Emp, name):
    """
    Entrena y evalúa un modelo de regresión XGBoost para una serie de tiempo.

    Esta función realiza un flujo de trabajo de entrenamiento estándar para un
    modelo de aprendizaje automático. El proceso incluye la división de los datos
    en conjuntos de entrenamiento y prueba, la configuración de los parámetros
    del modelo XGBoost, el entrenamiento del modelo y la evaluación del
    rendimiento.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame original de la serie de tiempo. Aunque se pasa como argumento,
        no se utiliza directamente en la lógica de la función, ya que el
        entrenamiento se realiza con `df_features`.
    df_features : pandas.DataFrame
        El DataFrame que contiene las variables predictoras (características) y
        la variable objetivo. Se espera que este DataFrame ya esté preparado.
    Emp : str
        Parámetro que se espera que sea un identificador de la empresa, pero
        no se utiliza en la lógica actual de la función.
    name : str
        El nombre de la columna que contiene la variable objetivo a predecir.

    Retorna:
    --------
    float
        El valor del coeficiente de determinación (R^2) del modelo en el
        conjunto de prueba, que indica la proporción de la varianza en la
        variable objetivo que es predecible a partir de las variables
        predictoras.
    """
    # Se crea una copia del DataFrame de entrada. La variable `df_aux` no se utiliza
    # en la lógica de entrenamiento.
    df_aux = df.copy()

    ### -------------------------------------------------------------------------------------------------------- ###
    #                                                    # ENTRENAMIENTO
    ### -------------------------------------------------------------------------------------------------------- ###

    # Se divide el DataFrame de características en conjuntos de entrenamiento y prueba.
    # El 80% de los datos se utiliza para entrenar el modelo y el 20% restante
    # para evaluar su rendimiento.
    split_idx = int(len(df_features) * 0.8)
    train = df_features.iloc[:split_idx]
    test = df_features.iloc[split_idx:]

    # --- Paso 3: Separar variables predictoras y objetivo ---
    # Se dividen los datos de entrenamiento y prueba en características (X) y
    # la variable objetivo (y).
    X_train = train.drop(name, axis=1)
    y_train = train[name]
    X_test = test.drop(name, axis=1)
    y_test = test[name]

    # Se definen los parámetros para el modelo XGBoost. Estos hiperparámetros
    # controlan la complejidad y el rendimiento del modelo.
    params_xgboost = {
        'objective': 'reg:squarederror',  # Objetivo de regresión para minimizar el error cuadrático.
        'n_estimators': 1000,             # El número de árboles de decisión a construir.
        'learning_rate': 0.05,            # Tasa de aprendizaje, controla el peso de cada nuevo árbol.
        'max_depth': 5,                   # La profundidad máxima de cada árbol.
        'subsample': 0.8,                 # La fracción de datos a muestrear para cada árbol.
        'colsample_bytree': 0.8,          # La fracción de características a muestrear para cada árbol.
        'random_state': 42,               # Semilla para la reproducibilidad de los resultados.
        'n_jobs': -1                      # Utiliza todos los núcleos de la CPU disponibles para el entrenamiento.
    }
    
    # 2. Configuración y entrenamiento del modelo XGBoost
    # Se instancia el modelo XGBoost con los parámetros definidos.
    modelo_xgb = xgb.XGBRegressor(**params_xgboost)
    
    # Entrenar el modelo con los datos de entrenamiento.
    # `eval_set` permite monitorear el rendimiento en el conjunto de prueba durante el entrenamiento.
    # `verbose=False` desactiva la impresión de resultados durante el entrenamiento.
    modelo_xgb.fit(X_train, y_train,
                   eval_set=[(X_train, y_train), (X_test, y_test)],
                   verbose=False)

    # 3. Realizar predicciones
    # Se realizan predicciones en el conjunto de prueba.
    predicciones_finales = modelo_xgb.predict(X_test)

    # Evaluar el modelo
    # Se calculan las métricas de error:
    # - RMSE (Raíz del Error Cuadrático Medio): Mide la magnitud de los errores.
    # - MAE (Error Absoluto Medio): Mide la magnitud promedio de los errores sin signo.
    # - R^2 (Coeficiente de Determinación): Mide qué tan bien el modelo predice la variabilidad.
    rmse = np.sqrt(mean_squared_error(y_test, predicciones_finales))
    mae = mean_absolute_error(y_test, predicciones_finales)
    r2 = r2_score(y_test, predicciones_finales)

    # Se retorna el valor de R^2.
    return r2
####  -------------------------------------------------------------------------------------  
#           Entrenamiento del modelo  
####  -------------------------------------------------------------------------------------

def Entrenamiento(df, df_features, Emp, name):
    """
    Entrena, evalúa y visualiza el rendimiento de un modelo de regresión XGBoost.

    Esta función realiza un flujo de trabajo completo para un modelo de
    aprendizaje automático. Primero, divide los datos en conjuntos de
    entrenamiento y prueba. Luego, entrena un modelo XGBoost con
    hiperparámetros fijos. Finalmente, evalúa el modelo, imprime
    la importancia de las características y genera varios gráficos para
    comparar las predicciones con los valores reales.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame original de la serie de tiempo. No se utiliza
        directamente en el entrenamiento, pero se pasa como argumento.
    df_features : pandas.DataFrame
        El DataFrame que contiene las variables predictoras (características)
        y la variable objetivo.
    Emp : str
        El nombre de la empresa o entidad, utilizado para imprimir mensajes
        y títulos de gráficos.
    name : str
        El nombre de la columna que es la variable objetivo a predecir.

    Retorna:
    --------
    modelo_xgb : xgb.XGBRegressor
        El modelo XGBoost entrenado.
    """
    # Imprime un separador y el encabezado del proceso.
    print(100 * '-')
    print("Modelo para " + name + ' de ' + Emp)
    
    # Crea una copia del DataFrame de entrada. La variable `df_aux` no se usa.
    df_aux = df.copy()

    # ### -------------------------------------------------------------------------------------------------------- ###
    # #                                                     # ENTRENAMIENTO
    # ### -------------------------------------------------------------------------------------------------------- ###
    
    # Se divide el conjunto de datos en un 80% para entrenamiento y un 20% para prueba.
    split_idx = int(len(df_features) * 0.8)
    train = df_features.iloc[:split_idx]
    test = df_features.iloc[split_idx:]
    
    # --- Paso 3: Separar variables predictoras y objetivo ---
    # Se dividen los conjuntos de entrenamiento y prueba en X (características) y Y (objetivo).
    X_train = train.drop(name, axis=1)
    y_train = train[name]
    X_test = test.drop(name, axis=1)
    y_test = test[name]
    
    # Imprime un mensaje para indicar el inicio del entrenamiento.
    print(" ... Entrenando modelo XGBoost ...")
    
    # Se definen los hiperparámetros para el modelo XGBoost.
    params_xgboost = {
        'objective': 'reg:squarederror',  # Objetivo para problemas de regresión.
        'n_estimators': 1000,             # Número de árboles de decisión.
        'learning_rate': 0.05,            # Tasa de aprendizaje para reducir la contribución de cada árbol.
        'max_depth': 5,                   # Profundidad máxima de cada árbol.
        'subsample': 0.8,                 # Fracción de muestras a utilizar en cada árbol.
        'colsample_bytree': 0.8,          # Fracción de características a utilizar en cada árbol.
        'random_state': 42,               # Semilla para garantizar la reproducibilidad.
        'n_jobs': -1                      # Usa todos los núcleos del CPU para un entrenamiento más rápido.
    }
    
    # 2. Configuración y entrenamiento del modelo XGBoost
    # Se instancia el modelo XGBoost con los parámetros definidos.
    modelo_xgb = xgb.XGBRegressor(**params_xgboost)
    
    # Se entrena el modelo usando los datos de entrenamiento.
    # `eval_set` se usa para monitorear el rendimiento del modelo en los datos de prueba.
    # `verbose=False` desactiva la salida de entrenamiento para que sea más limpio.
    modelo_xgb.fit(X_train, y_train,
                   eval_set=[(X_train, y_train), (X_test, y_test)],
                   verbose=False)
    
    # 3. Realizar predicciones
    # Se hacen predicciones en el conjunto de prueba.
    predicciones_finales = modelo_xgb.predict(X_test)

    # Evaluar el modelo y calcular métricas de rendimiento.
    rmse = np.sqrt(mean_squared_error(y_test, predicciones_finales))
    mae = mean_absolute_error(y_test, predicciones_finales)
    r2 = r2_score(y_test, predicciones_finales)
    
    # Calcular la importancia de cada característica para el modelo.
    importancias = modelo_xgb.feature_importances_
    nombres_caracteristicas = X_train.columns
    
    # Crear un DataFrame para mostrar la importancia de las características de manera clara.
    importancia_df = pd.DataFrame({'Caracteristica': nombres_caracteristicas,
                                   'Importancia': importancias})
    
    # Ordenar las características por importancia de mayor a menor.
    importancia_df = importancia_df.sort_values(by='Importancia', ascending=False)
    
    # ### -------------------------------------------------------------------------------------------------------- ###
    # #                                                    # GRÁFICOS
    # ### -------------------------------------------------------------------------------------------------------- ###
    
    # Gráfico 1: Comparación de valores verdaderos vs. predicciones.
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test.values, label='Valores Verdaderos', color='blue', linestyle='-', linewidth=2)
    plt.plot(y_test.index, predicciones_finales, label='Predicciones del Modelo', color='red', linestyle='--', linewidth=2)
    plt.title(f'Comparación de Valores Verdaderos vs. Predicciones para {Emp} con R² = {r2:.2f}', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show() # 
    
    # Gráfico 2: Comparación con los datos de entrenamiento.
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test.values, label='Valores Verdaderos', color='blue', linestyle='-', linewidth=2)
    plt.plot(y_test.index, predicciones_finales, label='Predicciones del Modelo', color='red', linestyle='--', linewidth=2)
    plt.plot(y_train.index, y_train.values, label='valores de entrenamiento ', color='black', linestyle=':', linewidth=2)
    plt.title(f'Comparación de Valores Verdaderos vs. Predicciones para {Emp} con R² = {r2:.2f}', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show() # 
    
    # Gráfico 3: Gráfico de dispersión para visualizar la relación entre los valores reales y las predicciones.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predicciones_finales, alpha=0.6)
    plt.title(f'Valores Verdaderos vs. Predicciones (Dispersión) para {Emp} con R² = {r2:.1f}', fontsize=16)
    plt.xlabel('Valores Verdaderos', fontsize=12)
    plt.ylabel('Predicciones', fontsize=12)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2, color='red')
    plt.grid(True)
    plt.show() # 
    
    # Imprime un separador al final del proceso.
    print(100 * '-')
    
    # Retorna el modelo entrenado.
    return modelo_xgb