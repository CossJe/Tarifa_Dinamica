# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:40:26 2025

@author: Jesus Coss
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import statsmodels.api as sm
from datetime import timedelta
from src.dynamic_pricing_data_loader import cargar_y_preparar_datos

def Get_goodDay(df, atribute, atribute1):
    """
    Identifica los 3 días de la semana con la mayor suma en un atributo
    y crea una columna indicadora.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada que contiene los datos.
    atribute : str
        Nombre de la columna numérica para la cual se calculará el total
        por día de la semana (e.g., 'VENTA_BOLETOS').
    atribute1 : str
        Nombre de la columna de fecha (que será convertida a datetime).

    Retorna:
    --------
    pd.DataFrame
        El DataFrame original con una nueva columna 'Buen_dia' (1 si es
        uno de los 3 días principales, 0 si no lo es).
    """
    # Se crea una copia del DataFrame para evitar modificar el original.
    df_output = df.copy()

    # Se asegura que la columna de fecha sea de tipo datetime.
    df_output[atribute1] = pd.to_datetime(df_output[atribute1])

    # Se extrae el nombre del día de la semana.
    df_output['Dia de la semana'] = df_output[atribute1].dt.day_name()
    
    # Se calcula la suma del atributo por cada día de la semana.
    daily_sales = df_output.groupby('Dia de la semana')[atribute].sum()
    
    # Se ordenan los días por ventas de forma descendente y se toman los 3 primeros.
    daily_sales = daily_sales.sort_values(ascending=False)
    daily_sales = daily_sales[:3]

    # Se crea un diccionario para mapear los nombres de los días a números (0-6).
    indices = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
               
    # Se obtienen los índices numéricos de los 3 días principales.
    dias = list(daily_sales.index.map(indices).to_numpy())
    
    # Se crea la columna 'Buen_dia' (1 si el día de la semana está en la lista de "días buenos", 0 si no).
    df_output['Buen_dia'] = df_output[atribute1].dt.dayofweek.isin(dias).astype(int)
    
    # Se asegura que la columna 'Buen_dia' sea de tipo entero de 64 bits.
    df_output['Buen_dia'] = df_output['Buen_dia'].astype(np.int64)
    
    # Se elimina la columna auxiliar 'Dia de la semana'.
    df_output.drop('Dia de la semana', axis=1, inplace=True)
    
    return df_output

def GetData(Bandera):
    """
    Carga, filtra y prepara los datos de ventas para un año específico.

    La función lee un archivo, selecciona las columnas relevantes, filtra
    los datos por ventas positivas y por un año determinado, y realiza
    cálculos adicionales de precios y tarifas.

    Parámetros:
    -----------
    year : int
        El año para el cual se desea filtrar los datos.

    Retorna:
    --------
    pd.DataFrame
        El DataFrame limpio y preparado con los datos de ventas para el año
        especificado.
    """
    # Se definen las rutas de los archivos de configuración y datos.
    ruta_principal = os.getcwd()
    config_path = os.path.join(ruta_principal, "config", "config.json")
    
    # Se cargan y preparan los datos usando una función externa.
    Frame = cargar_y_preparar_datos(config_path, ruta_principal)
    
    # Se seleccionan las columnas de interés.
    Df = Frame[['OPERACION', 'FECHA_CORRIDA', 'FECHA_OPERACION', 'TIPO_PASAJERO', 'VENTA',
                'KMS_TRAMO', 'PAX_SUBEN', 'VENTA_ANTICIPADA', 'TIPO_CORRIDA','HORAS_ANTICIPACION',
                'DESC_DESCUENTO', 'PORCENT_PROMO', 'IVA_VENDIDO', 'TARIFA_BASE_TRAMO', 'IVA_TARIFA_BASE_TRAMO',
                'HORA_SALIDA_CORRIDA']]

    # Se filtra el DataFrame para incluir solo ventas mayores que cero.
    Df = Df[Df['VENTA'] > 0]
    
    # Se convierte la columna 'FECHA_OPERACION' a tipo datetime.
    Df['FECHA_OPERACION'] = pd.to_datetime(Df['FECHA_OPERACION'])
    
    if Bandera:
        # Se filtra por el año especificado.
        Df = Df[Df['FECHA_OPERACION'].dt.year == 2024]
    else:
        # 1. Encontrar la fecha máxima
        fecha_maxima = Df['FECHA_OPERACION'].max()
        
        # 2. Calcular el día anterior a la fecha máxima
        dia_anterior = fecha_maxima - timedelta(days=1)
        
        # 3. Calcular la fecha de inicio (365 días antes del día_anterior)
        fecha_inicio = dia_anterior - timedelta(days=364) # Para incluir 365 días, es decir, día_anterior y los 364 previos.
        
        # 4. Filtrar el DataFrame
        # Se incluyen todas las fechas en el rango [fecha_inicio, dia_anterior]
        Df = Df[
            (Df['FECHA_OPERACION'] >= fecha_inicio) & 
            (Df['FECHA_OPERACION'] <= dia_anterior)
        ].copy()
    
    # Se calculan las columnas 'PRECIO' y 'TARIFA' restando el IVA.
    Df['PRECIO'] = Df['VENTA'] + Df['IVA_VENDIDO']
    Df['TARIFA'] = Df['TARIFA_BASE_TRAMO'] + Df['IVA_TARIFA_BASE_TRAMO']
    
    # Se vuelven a seleccionar las columnas finales.
    Df = Df[['OPERACION', 'FECHA_CORRIDA', 'FECHA_OPERACION', 'TIPO_PASAJERO', 'VENTA', 'PRECIO',
             'KMS_TRAMO', 'PAX_SUBEN', 'VENTA_ANTICIPADA', 'TIPO_CORRIDA', 'TARIFA',
             'DESC_DESCUENTO', 'PORCENT_PROMO','HORAS_ANTICIPACION','HORA_SALIDA_CORRIDA']]
    
    # Se reemplazan los valores NaN en 'PORCENT_PROMO' con 0.
    Df['PORCENT_PROMO'] = Df['PORCENT_PROMO'].fillna(0)
    
    # Se retorna solo las filas donde 'TIPO_CORRIDA' es 'NORMAL'.
    return Df[Df['TIPO_CORRIDA'] == 'NORMAL']

def GetFeatures(df,BC_json):
    """
    Realiza la ingeniería de características en un DataFrame de datos de ventas.

    Esta función calcula variables como la anticipación de la compra, el mes,
    si es fin de semana y si es un "día bueno" de ventas. También renombra
    una columna y elimina filas con valores nulos.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada con los datos crudos de ventas.

    Retorna:
    --------
    pd.DataFrame
        El DataFrame con las nuevas características generadas y sin valores nulos.
    """
    # Se crea una copia del DataFrame para evitar modificar el original.
    Frame = df.copy()

    # Se convierten las columnas de fecha a tipo datetime.
    Frame['FECHA_OPERACION'] = pd.to_datetime(Frame['FECHA_OPERACION'])
    Frame['FECHA_CORRIDA'] = pd.to_datetime(Frame['FECHA_CORRIDA'])
    
    # Se crea la columna 'Dias_Anticipacion'.
    Frame['Dias_Anticipacion'] = (Frame['FECHA_CORRIDA'] - Frame['FECHA_OPERACION']).dt.days
    
    # Se extraen el mes y se crea una bandera para el fin de semana.
    Frame['Mes_Viaje'] = Frame['FECHA_CORRIDA'].dt.month
    Frame['Fin_Semana_Viaje'] = Frame['FECHA_CORRIDA'].dt.dayofweek >= 5
    
    # Se renombra la columna 'PAX_SUBEN' a 'Q_Boletos'.
    Frame.rename(columns={'PAX_SUBEN': 'Q_Boletos'}, inplace=True)
    
    # Se llama a la función Get_goodDay para agregar la columna 'Buen_dia'.
    #Frame = Get_goodDay(Frame, 'PRECIO', 'FECHA_CORRIDA')
    
    # Se convierten las columnas booleanas a tipo entero.
    #Frame['Fin_Semana_Viaje'] = Frame['Fin_Semana_Viaje'].astype(int)
    #Frame['Buen_dia'] = Frame['Buen_dia'].astype(int)
    
    Frame['Buen_Dia'] = Frame['FECHA_CORRIDA'].dt.dayofweek.isin(BC_json["DiaBueno"]).astype(int)
    Frame['Buena_Hora'] = Frame['HORA_SALIDA_CORRIDA'].dt.hour.isin(BC_json["HoraBuena"]).astype(int)
    Frame['Buen_Mes'] = Frame['FECHA_CORRIDA'].dt.month.isin(BC_json["MesBueno"]).astype(int)
    
    # Se eliminan las filas con valores nulos.
    Frame = Frame.dropna()
    
    return Frame


def GetElasticity_Log(df):
    """
    Calcula la elasticidad precio de la demanda usando regresión log-log en dos etapas.

    Parámetros:
    -----------
    df : pd.DataFrame
        Debe contener 'PRECIO', 'Q_Boletos', 'TARIFA' y otras variables exógenas.

    Retorna:
    --------
    tuple
        - Coef (list): Coeficientes de la segunda etapa.
        - elasticidad (float): Elasticidad precio de la demanda.
    """
    Frame = df.copy()
    
    # Transformar precio y cantidad a logaritmo
    Frame['log_PRECIO'] = np.log(Frame['PRECIO'])
    Frame['log_Q_Boletos'] = np.log(Frame['Q_Boletos'])
    
    # Primera etapa: regresión del log-precio sobre variables instrumentales
    X_P = Frame[['TARIFA', 'HORAS_ANTICIPACION', 'Buen_Mes', 'Buena_Hora', 'Buen_Dia']]
    X_P = sm.add_constant(X_P)
    Y_P = Frame['log_PRECIO']
    
    reg_stage1 = sm.OLS(Y_P, X_P).fit()
    
    # Valores predichos del log-precio
    T_Est = pd.DataFrame(reg_stage1.predict(X_P), columns=['log_T_Est'])
    
    # Segunda etapa: regresión de log-cantidad sobre log-precio estimado y exógenas
    X_Q = Frame[['HORAS_ANTICIPACION', 'Buen_Mes', 'Buena_Hora', 'Buen_Dia']]
    X_Q = pd.concat([T_Est, X_Q], axis=1)
    X_Q = sm.add_constant(X_Q)
    
    Y_Q = Frame['log_Q_Boletos']
    reg_stage2 = sm.OLS(Y_Q, X_Q).fit()
    
    # Elasticidad precio directamente como coeficiente del log-precio estimado
    beta1 = reg_stage2.params['log_T_Est']
    elasticidad = beta1  # En log-log, el coeficiente ya es la elasticidad
    
    print(f"La elasticidad de la demanda (log-log) es: {elasticidad:.4f}")
    
    Coef = list(reg_stage2.params)
    
    return Coef, elasticidad


def GetElasticity(df):
    """
    Calcula la elasticidad precio de la demanda utilizando regresión en dos etapas.

    Este método de variables instrumentales corrige la endogeneidad del precio
    (la correlación entre el precio y el término de error) al usar la tarifa
    ofrecida como un instrumento para el precio final de venta.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos limpios y las características. Debe contener
        'PRECIO', 'Q_Boletos', 'TARIFA' y otras variables.

    Retorna:
    --------
    tuple
        - Coef (list): Lista de los coeficientes del modelo de la segunda etapa.
        - elasticidad (float): El valor calculado de la elasticidad de la demanda.
    """
    Frame = df.copy()
    
    # Se definen las variables para la primera etapa de la regresión.
    # X_P son las variables predictoras del precio de venta, incluyendo la tarifa.
    X_P = Frame[['TARIFA', 'HORAS_ANTICIPACION', 'Buen_Mes', 'Buena_Hora', 'Buen_Dia']]
    Y_P = pd.DataFrame(Frame['PRECIO'])
    
    # Se definen las variables para la segunda etapa.
    X_Q = Frame[['HORAS_ANTICIPACION', 'Buen_Mes', 'Buena_Hora', 'Buen_Dia']]
    
    # Primera Etapa: Regresión del Precio de Venta (Y_P) sobre sus predictoras (X_P).
    X_P = sm.add_constant(X_P)  # Se agrega una constante para el intercepto.
    reg_stage1 = sm.OLS(Y_P, X_P).fit()
    
    # Se obtienen los valores predichos del precio de venta (T_Est).
    T_Est = reg_stage1.predict(X_P)
    T_Est = pd.DataFrame(T_Est, columns=['T_Est'])
    
    # Segunda Etapa: Regresión de la Cantidad de Boletos (Y_Q) sobre el precio estimado (T_Est)
    # y otras variables exógenas (X_Q).
    X_Q = pd.concat([T_Est, X_Q], axis=1)
    X_Q = sm.add_constant(X_Q)  # Se agrega una constante para el intercepto.
    Y_Q = pd.DataFrame(Frame['Q_Boletos'])
    
    reg_stage2 = sm.OLS(Y_Q, X_Q).fit()
    
    # Se obtiene el coeficiente beta1, que es el coeficiente del precio estimado.
    beta1 = reg_stage2.params['T_Est']
    
    # Se calculan los promedios del precio y la cantidad de boletos.
    promedio_Q_Boletos = Frame['Q_Boletos'].mean()
    promedio_P_Venta = Frame['PRECIO'].mean()
    
    # Se calcula la elasticidad de la demanda con la fórmula de elasticidad puntual.
    elasticidad = beta1 * (promedio_P_Venta / promedio_Q_Boletos)
    
    print(f"La elasticidad de la demanda es: {elasticidad:.4f}")
    
    # Se extraen los coeficientes del modelo de la segunda etapa.
    Coef = list(reg_stage2.params)
    
    return Coef, elasticidad

def GetPrizes(Coef, CondIni, UT):
    """
    Calcula un precio máximo y un precio sugerido de venta.

    La función utiliza los coeficientes de una regresión de elasticidad de la
    demanda para estimar precios que optimicen los ingresos o se ajusten a
chos de venta.

    Parámetros:
    -----------
    Coef : list
        Una lista de los coeficientes de la regresión de la elasticidad
        (obtenidos de la función GetElasticity).
    CondIni : dict
        Un diccionario con las condiciones iniciales para el pronóstico,
        como los días de anticipación, el mes, etc.
    UT : float
        Un factor de utilidad o umbral (utility threshold) que afecta
        el precio sugerido.

    Retorna:
    --------
    tuple
        - PrecioMaximo (float): El precio máximo calculado.
        - PrecioSugerido (float): El precio sugerido ajustado.
    """
    # Se calcula un "precio máximo" inicial usando los coeficientes
    # y las condiciones iniciales. Este precio representa la intersección
    # de la curva de la demanda con el eje de cantidad cero.
    PrecioMaximo = Coef[0] + Coef[2] * CondIni['Dias_Anticipacion'] + \
                   Coef[3] * CondIni['Mes_Viaje'] + Coef[4] * CondIni['Fin_Semana_Viaje'] + \
                   Coef[5] * CondIni['Buen_dia']
    
    # Se calcula el punto de máxima utilidad (PM) y un "precio de saturación" (PS)
    # utilizando los coeficientes del modelo.
    PM = np.abs(PrecioMaximo / (-2 * Coef[1]))
    PS = np.abs(Coef[0] / (Coef[1] * 2))
    
    # Se calcula el factor de ajuste Dp (utility delta).
    Dp = (UT - PM) / UT
    
    # Se ajusta el PrecioMaximo y el PrecioSugerido usando el factor Dp.
    PrecioMaximo = Dp * PM + PM
    PrecioSugerido = Dp * PS + PS
    
    #print(f"El precio maximo a vender es: {PrecioMaximo:.4f}")

    return PrecioMaximo, PrecioSugerido


def MainElas():
    """
    Función principal para calcular la elasticidad de la demanda y sugerir precios.

    Esta función orquesta un flujo de trabajo que incluye la carga de datos,
    la ingeniería de características, el cálculo de la elasticidad de la demanda
    y la sugerencia de precios óptimos.

    Parámetros:
    -----------
    UltimaTar : float, opcional
        La última tarifa conocida, utilizada como un valor de referencia para
        los cálculos de precios. Por defecto es 1163.79.

    Retorna:
    --------
    tuple
        - PrecioMaximo (float): El precio máximo sugerido para la venta.
        - PrecioSugerido (float): El precio sugerido para optimizar los ingresos.
        - Elasticidad (float): El valor de la elasticidad precio de la demanda.
    """
    ruta_principal = os.getcwd()
    
    # Se definen las condiciones iniciales para el pronóstico.
    CondIni = {
        'Dias_Anticipacion': 0,
        'Mes_Viaje': 12,
        'Fin_Semana_Viaje': 0,
        'Buen_dia': 1
    }
    
    BuenasCarac_path = os.path.join(ruta_principal, "Files", "BuenaCaracteristicas.json")
    with open(BuenasCarac_path, 'r') as f:
        # 2. Cargar el contenido del archivo JSON
        BC_json = json.load(f)
            
    # Se obtienen y preparan los datos para el año 2024.
    Frame = GetData(False)
    TBT= Frame['TARIFA'].iloc[-1]
    Frame = GetFeatures(Frame,BC_json)
    
    # Se calculan los coeficientes del modelo de elasticidad y el valor de la elasticidad.
    Coef, Elasticidad = GetElasticity(Frame)
    
    # Se utilizan los coeficientes y las condiciones iniciales para sugerir precios.
    PrecioM, PrecioS = GetPrizes(Coef, CondIni, TBT)
    
    # Convertir a tipos nativos de Python
    data = {
        "TBT": int(TBT),
        "PrecioMaximo": float(PrecioS),
        "PrecioSugerido": float(PrecioM),
        "Elasticidad": float(Elasticidad)
    }
    
    config_path = os.path.join(ruta_principal, "Files", "Resultados_Elasticidad.json")
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return 

def GetDataElasticity():
    ruta_principal = os.getcwd()
    config_path = os.path.join(ruta_principal, "Files", "Resultados_Elasticidad.json")
    try:
        with open(config_path, 'r') as f:
            # 2. Cargar el contenido del archivo JSON
            datos_json = json.load(f)
            
        
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta: {config_path}")
    except json.JSONDecodeError:
        print("Error: El archivo JSON tiene un formato inválido.")
        
    return datos_json
    
