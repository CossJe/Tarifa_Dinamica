# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:11:26 2025

@author: Jesus Coss
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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
    
def predicciones(df, df1, carac, modelo_xgb, n_futuro=1):
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
    df1 : pandas.DataFrame
        Un DataFrame externo que contiene una variable predictora para las
        fechas futuras.
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
    tuple
        - pandas.DataFrame: El DataFrame original concatenado con las filas de
          predicciones futuras.
        - datetime.Timestamp: La última fecha del DataFrame original.
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

        # Se agrega el valor de la variable externa (`df1`) en el punto
        # de tiempo de la predicción. Esto podría causar "data leakage" si
        # la variable no se conoce en el momento de la predicción.
        df_Pred['VENTA'] = df1.loc[future_dates[i]][df1.columns[0]]
        
        # Se eliminan los valores objetivo de las características para la predicción.
        X_future = df_Pred.drop(df_Pred.columns[0], axis=1)
        
        # El modelo realiza la predicción.
        prediccion = modelo_xgb.predict(X_future)
        
        # Se inserta el valor predicho en el DataFrame principal,
        # lo que permite que sea utilizado como lag en la próxima iteración.
        df_forecast_full.loc[future_dates[i], df_forecast_full.columns[0]] = prediccion[0]
        
    # Se retorna el DataFrame completo con los valores pronosticados y la última fecha.
    return df_forecast_full, last_date

    
def Obtener_Carac_Futuro(df_SCAaP, carac):
    """
    Generates features for the next point in a time series.

    This function takes a DataFrame with historical data and a dictionary of
    feature parameters to create the same predictive variables used during
    model training. This allows a forecasting model (like XGBoost) to make a 
    prediction for the next period using the correct input features.

    Parameters:
    -----------
    df_SCAaP : pandas.DataFrame
        The input DataFrame containing the historical series of the variable 
        to be forecasted. (SCAaP = Solo Con el Atributo a Pronosticar - 
        "Only With the Attribute to be Forecasted").
    carac : dict
        A dictionary containing the parameters of the features (e.g., lags,
        Fourier terms) used to train the model.

    Returns:
    --------
    pandas.DataFrame
        A single-row DataFrame containing all the necessary features for the
        next period's prediction.
    """
    # Create a copy of the input DataFrame.
    df_4lags = df_SCAaP.copy()

    # Add standard lags and their 2-day rolling means.
    for lag in carac['lags']:
        df_4lags[f'lag_{lag}'] = df_4lags[df_4lags.columns[0]].shift(lag)
        df_4lags[f'Rolling_Mean_lag_{lag}'] = (
            df_4lags[df_4lags.columns[0]].shift(lag).rolling(window=2).mean())

    # Isolate the last row to create the features for the future.
    df = df_4lags.iloc[-1:].copy()
    
    # This auxiliary DataFrame is created but not used in the subsequent logic.
    df_4Componentes = pd.DataFrame(df_4lags.iloc[:-1, 0])

    # Add day-of-week and "good day" features for the last row.
    df['Dia de la semana'] = df.index.dayofweek
    df['Dia de la semana'] = df['Dia de la semana'].astype(np.int64)
    df['Es buen dia'] = df.index.dayofweek.isin(carac['dias']).astype(int)

    # Create a new auxiliary DataFrame to calculate day-of-week specific lags.
    df_4lags = df_SCAaP.copy()
    df_4lags['Dia de la semana'] = df_4lags.index.dayofweek
    df_4lags['Dia de la semana'] = df_4lags['Dia de la semana'].astype(np.int64)

    # Initialize columns for weekly lags with NaNs.
    for lag in [7, 14, 21]:
        df_4lags[f'{df_4lags.columns[0]}_lag_{lag}'] = np.nan
        
    # Calculate day-of-week specific lags.
    for day in range(7):
        # Filter the DataFrame for a specific day of the week.
        subset_dia = df_4lags[df_4lags['Dia de la semana'] == day].copy()
        
        # Calculate lags for that subset.
        subset_dia[f'{df_4lags.columns[0]}_lag_7'] = subset_dia[df_4lags.columns[0]].shift(1)
        subset_dia[f'{df_4lags.columns[0]}_lag_14'] = subset_dia[df_4lags.columns[0]].shift(2)
        subset_dia[f'{df_4lags.columns[0]}_lag_21'] = subset_dia[df_4lags.columns[0]].shift(3)
        
        # Update the main DataFrame with the calculated lags for that day.
        df_4lags.update(subset_dia)
        
    # Extract the last values of the weekly lags and assign them to the output DataFrame.
    for lag in [7, 14, 21]:
        df[f'{df.columns[0]}_lag_{lag}'] = df_4lags[f'{df_4lags.columns[0]}_lag_{lag}'].iloc[-1]
        
    # Add a feature indicating if the next day will be above average using VaR.
    df['DAP'] = (VaR(df_4lags, df_4lags.columns[0]) >= df_4lags[df_4lags.columns[0]][:-1].mean())
    
    # Extract more date-based features.
    df['dia_del_mes'] = df.index.day
    df['dia_del_año'] = df.index.dayofyear
    df['semana_del_año'] = df.index.isocalendar().week.astype(int)
    
    # Add Fourier terms based on parameters from the `carac` dictionary.
    freq = 7
    for i in range(1, carac['n_terms'] + 1, carac['n']):
        df[f'sin_{freq}_{i}'] = np.sin(2 * np.pi * i * df.index.dayofweek / 7)
        df[f'cos_{freq}_{i}'] = np.cos(2 * np.pi * i * df.index.dayofweek / 7)
        
    # Add an outlier risk feature.
    df['es_outlier'] = riesgo_outlier_df(df_4lags, df_4lags.columns[0], 7, carac['rate'])
    
    # Return the final DataFrame with all features for prediction.
    return df