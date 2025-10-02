# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:09:34 2025

@author: Jesus Coss
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

####  -------------------------------------------------------------------------------------  
#           Predicciones 
####  -------------------------------------------------------------------------------------
def riesgo_outlier_df(df, column, ventana=7, umbral_vol=1.5):
    """
    Evalúa si hay riesgo de que el próximo valor de una serie (en un DataFrame)
    sea un outlier, usando volatilidad local y tendencia de la media móvil.
    
    Esta función analiza el comportamiento reciente de una serie de tiempo
    para predecir la probabilidad de que el próximo punto de datos sea un
    valor atípico. Lo hace combinando dos métricas clave: la volatilidad
    local (desviación estándar móvil) y un cambio significativo en la
    tendencia de la media móvil.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con la serie de tiempo histórica. El índice debe ser
        cronológico para un cálculo correcto de la ventana móvil.
    column : str
        Nombre de la columna a analizar para detectar el riesgo de outlier.
    ventana : int, opcional
        El tamaño de la ventana (número de periodos) para calcular la
        desviación estándar móvil y la media móvil. Por defecto es 7.
    umbral_vol : float, opcional
        El factor multiplicador que define un umbral para considerar la
        volatilidad como alta. Por defecto es 1.5, lo que significa que
        la volatilidad reciente debe ser 50% mayor que el promedio.

    Retorna:
    --------
    int
        Un valor booleano (convertido a entero) que indica si hay riesgo
        de outlier. Retorna 1 si el riesgo está presente (True), y 0 en
        caso contrario (False).
    """
    # Se extrae la columna de la serie de tiempo del DataFrame.
    serie = df[column]
    
    # Se calcula la desviación estándar móvil (volatilidad local).
    std_movil = serie.rolling(window=ventana).std()
    
    # Se evalúa si la última volatilidad es significativamente mayor que el promedio.
    riesgo_vol = std_movil.iloc[-1] > std_movil.mean() * umbral_vol
    
    # Se calcula la media móvil.
    media_movil = serie.rolling(window=ventana).mean()
    
    # Se calcula la tendencia como la diferencia entre medias móviles consecutivas.
    tendencia = media_movil.diff()
    
    # Se evalúa si el último cambio en la tendencia es grande en comparación con la variación histórica.
    riesgo_tendencia = abs(tendencia.iloc[-1]) > tendencia.std()
    
    # El riesgo total es verdadero si se cumple alguna de las dos condiciones de riesgo.
    riesgo_total = bool(riesgo_vol or riesgo_tendencia)
    
    # Se convierte el resultado booleano a entero (0 o 1) para una fácil integración en modelos.
    riesgo_total = int(riesgo_total)
    
    return riesgo_total

def VaR(df, column):
    """
    Calcula el Valor en Riesgo (VaR) paramétrico para una serie de precios.

    Esta función estima la pérdida potencial máxima en un día (t=1)
    utilizando una volatilidad calculada con un modelo de media móvil
    exponencialmente ponderada (EWMA). El VaR se calcula a un nivel de
    confianza del 95%.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene la serie de precios.
    column : str
        El nombre de la columna que contiene los precios.

    Retorna:
    --------
    float
        El valor del activo en el que se esperaría que la pérdida máxima
        no exceda el VaR. Esto es equivalente al precio actual menos el VaR.
    """
    # Se define el horizonte de tiempo (1 día) y el nivel de confianza.
    t = 1
    nivel_confianza = 0.95
    
    # Se calcula el Z-Score de la distribución normal estándar para el nivel de confianza.
    # Por ejemplo, para el 95%, el valor es aproximadamente -1.645.
    z = norm.ppf(1 - nivel_confianza)
    
    # Se extrae la serie de precios. `[:-1]` se usa para ignorar el último valor si es NaN,
    # aunque no está claro por qué se hace en este contexto.
    serie = df[column][:-1].values
    
    # Se toma el último valor de la serie como el precio actual (o precio inicial del período de VaR).
    M = serie[-1]
    
    # Se calculan los rendimientos logarítmicos.
    # `serie[:-1]/serie[1:]` calcula el rendimiento entre cada punto.
    Rent = np.log(serie[:-1] / serie[1:])
    
    # Se elevan al cuadrado los rendimientos para obtener los rendimientos cuadrados.
    Rent2 = Rent ** 2
    
    # Se define el factor de decaimiento (lambda) para el modelo EWMA.
    # Un valor de 0.94 es el estándar de la industria (RiskMetrics).
    lamda = 0.94
    
    # Se crean los pesos para cada rendimiento, dando más peso a los valores recientes.
    pos = np.arange(len(Rent), 0, -1)
    ponderacion = (1 - lamda) * lamda ** (pos - 1)
    
    # Se calcula la varianza ponderada de los rendimientos.
    varianza = ponderacion * Rent2
    
    # La volatilidad es la raíz cuadrada de la suma de las varianzas ponderadas.
    volatilidad = np.sqrt(np.sum(varianza))
    
    # Se calcula el VaR paramétrico utilizando la fórmula:
    # VaR = Precio * Volatilidad * Raíz(t) * Z-Score
    VaR_Par = M * volatilidad * np.sqrt(t) * z
    
    # Se calcula el valor esperado del activo después de aplicar el VaR.
    # Es el valor actual del activo más el VaR (que es un valor negativo).
    Pax = M + VaR_Par
    
    # Se retorna el valor final.
    return Pax
    
def predicciones(df, carac, modelo_xgb, n_futuro=1):
    """
    Genera predicciones futuras para una serie de tiempo de forma iterativa.

    Esta función realiza pronósticos de un solo paso, utilizando las predicciones
    generadas como datos de entrada para las siguientes predicciones. Este
    método es ideal para modelos que dependen de valores pasados (lags)
    como características.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame que contiene la serie de tiempo histórica.
    carac : dict
        Un diccionario que contiene los parámetros de las características
        utilizados durante el entrenamiento del modelo.
    modelo_xgb : objeto
        El modelo de machine learning (XGBoost) entrenado.
    n_futuro : int, opcional
        El número de períodos (días) hacia el futuro que se desean predecir.
        Por defecto es 1.

    Retorna:
    --------
    pandas.DataFrame
        El DataFrame original concatenado con las filas de predicciones futuras.
    """
    # Se crea una copia del DataFrame de entrada.
    df_aux = df.copy()
    
    # Se obtiene la última fecha del DataFrame.
    last_date = df_aux.index.max()
    
    # Se genera un rango de fechas futuras para las predicciones.
    future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=n_futuro, freq='D')
    df_future = pd.DataFrame(index=future_dates)
    
    # Se concatenan los datos históricos con el DataFrame de fechas futuras vacío.
    df_forecast_full = pd.concat([df_aux, df_future])

    # Bucle para realizar predicciones iterativas.
    for i in range(n_futuro):
        # Se calcula el índice del corte del DataFrame para cada iteración.
        # Esto asegura que solo se utilicen los datos históricos y las
        # predicciones previas para generar la siguiente predicción.
        j = -n_futuro + i + 1
        
        # Se obtienen las características necesarias para la predicción.
        if j == 0:
            # En la primera iteración, se usan todos los datos históricos.
            df_Pred = Obtener_Carac_Futuro(df_forecast_full.iloc[:], carac)
        else:
            # En las siguientes, se usan los datos históricos más las
            # predicciones ya generadas.
            df_Pred = Obtener_Carac_Futuro(df_forecast_full.iloc[:j], carac)

        # Se eliminan los valores objetivo de las características para la predicción.
        X_future = df_Pred.drop(df_Pred.columns[0], axis=1)
        
        # El modelo realiza la predicción.
        prediccion = modelo_xgb.predict(X_future)
        
        # Se inserta el valor predicho en el DataFrame principal,
        # lo que permite que sea utilizado como lag en la próxima iteración.
        df_forecast_full.loc[future_dates[i], df_forecast_full.columns[0]] = prediccion[0]
        
    # Se retorna el DataFrame completo con los valores pronosticados.
    return df_forecast_full

    
def Obtener_Carac_Futuro(df_SCAaP, carac):
    """
    Genera características para el próximo punto de una serie de tiempo.

    Esta función toma un DataFrame con la serie histórica y un diccionario de
    parámetros de características (`carac`) para crear las mismas variables
    predictoras que se usaron en el entrenamiento. Esto permite que el
    modelo de pronóstico (generalmente XGBoost) pueda hacer una predicción
    para el siguiente período con la información adecuada.

    Parámetros:
    -----------
    df_SCAaP : pandas.DataFrame
        DataFrame que contiene la serie histórica de la variable a pronosticar.
        Su nombre (df Solo Con el Atributo a Pronosticar) indica que solo
        debe contener la variable objetivo.
    carac : dict
        Un diccionario que contiene los parámetros de las características
        (como los lags, la frecuencia y los términos de Fourier) que se
        usaron para entrenar el modelo.

    Retorna:
    --------
    pandas.DataFrame
        Un DataFrame de una sola fila que contiene todas las características
        necesarias para la predicción del siguiente período.
    """
    # Se crea una copia del DataFrame de entrada.
    df_4lags = df_SCAaP.copy()

    # Se agregan los lags y sus medias móviles de 2 días.
    for lag in carac['lags']:
        df_4lags[f'lag_{lag}'] = df_4lags[df_4lags.columns[0]].shift(lag)
        df_4lags[f'Rolling_Mean_lag_{lag}'] = (
            df_4lags[df_4lags.columns[0]].shift(lag).rolling(window=2).mean())

    # Se aísla la última fila del DataFrame para crear las características del futuro.
    df = df_4lags.iloc[-1:].copy()
    
    # La siguiente línea crea un DataFrame auxiliar, pero no se usa en la lógica posterior.
    df_4Componentes = pd.DataFrame(df_4lags.iloc[:-1, 0])

    # Se agregan características de fecha para la última fila.
    df['Dia de la semana'] = df.index.dayofweek
    df['Dia de la semana'] = df['Dia de la semana'].astype(np.int64)

    # Se agrega una característica binaria que indica si el día es considerado "bueno"
    # basándose en el diccionario de características.
    df['Es buen dia'] = df.index.dayofweek.isin(carac['dias']).astype(int)

    # Se crea un nuevo DataFrame auxiliar para calcular lags específicos por día de la semana.
    df_4lags = df_SCAaP.copy()
    df_4lags['Dia de la semana'] = df_4lags.index.dayofweek
    df_4lags['Dia de la semana'] = df_4lags['Dia de la semana'].astype(np.int64)

    # Se inicializan las columnas para los lags semanales con NaNs.
    for lag in [7, 14, 21]:
        df_4lags[f'{df_4lags.columns[0]}_lag_{lag}'] = np.nan
        
    # Se calculan los lags por día de la semana de forma iterativa.
    for day in range(7):
        # Se filtra el DataFrame para obtener solo los datos de un día de la semana específico.
        subset_dia = df_4lags[df_4lags['Dia de la semana'] == day].copy()
        
        # Se calculan los lags para ese subconjunto.
        subset_dia[f'{df_4lags.columns[0]}_lag_7'] = subset_dia[df_4lags.columns[0]].shift(1)
        subset_dia[f'{df_4lags.columns[0]}_lag_14'] = subset_dia[df_4lags.columns[0]].shift(2)
        subset_dia[f'{df_4lags.columns[0]}_lag_21'] = subset_dia[df_4lags.columns[0]].shift(3)
        
        # Se actualiza el DataFrame principal con los lags calculados para ese día.
        df_4lags.update(subset_dia)
        
    # Se extraen los últimos valores de los lags semanales calculados y se asignan al DataFrame de salida.
    for lag in [7, 14, 21]:
        df[f'{df.columns[0]}_lag_{lag}'] = df_4lags[f'{df_4lags.columns[0]}_lag_{lag}'].iloc[-1]
        
    # Se agrega una característica que indica si el próximo día estará por encima del promedio,
    # utilizando el cálculo del VaR.
    df['DAP'] = (VaR(df_4lags, df_4lags.columns[0]) >= df_4lags[df_4lags.columns[0]][:-1].mean())
    
    # Se extraen más características de la fecha.
    df['dia_del_mes'] = df.index.day
    df['dia_del_año'] = df.index.dayofyear
    df['semana_del_año'] = df.index.isocalendar().week.astype(int)
    
    # Se agregan los términos de Fourier según los parámetros en el diccionario `carac`.
    freq = 7
    for i in range(1, carac['n_terms'] + 1, carac['n']):
        df[f'sin_{freq}_{i}'] = np.sin(2 * np.pi * i * df.index.dayofweek / 7)
        df[f'cos_{freq}_{i}'] = np.cos(2 * np.pi * i * df.index.dayofweek / 7)
        
    # Se agrega una característica que evalúa el riesgo de que el próximo valor sea un outlier.
    df['es_outlier'] = riesgo_outlier_df(df_4lags, df_4lags.columns[0], 7, carac['rate'])
    
    # Se retorna el DataFrame final con todas las características para la predicción.
    return df