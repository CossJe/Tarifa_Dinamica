import pandas as pd
import numpy as np
import pyodbc
import os
import unicodedata

import warnings
warnings.filterwarnings('ignore')

#################################################################################################################
#### Funcion para leer datos de los modelos de DIVER
def DIVER_ODBC_Extraction( conect_pars, sql_query ):
    # Conectar a la base de datos
    conn = pyodbc.connect(f"\
                          DSN={ conect_pars['dsn'] };\
                          UID={ conect_pars['user'] };\
                          PWD={ conect_pars['pswd'] }\
                          ")
    # Cargar datos en un DataFrame de Pandas
    df = pd.read_sql( sql_query, conn )

    return df

#################################################################################################################
#### Función para leer datos de la base de datos DB2
def DB2ConnectorODBC(dsn=None, host=None, port=None, database=None, uid=None, pwd=None, driver="IBM DB2 ODBC DRIVER", sql=None):
    """
    Constructor para establecer los parámetros de conexión ODBC a Db2.

    - Si usas un DSN configurado, pasa solo 'dsn', 'uid' y 'pwd'.
    - Si no usas DSN, debes pasar: host, port, database, uid, pwd y opcionalmente driver.
    """
    connection_string = None
    connection = None
    df = None

    if dsn:
        connection_string = f"DSN={dsn};UID={uid};PWD={pwd};"
    elif all([host, port, database, uid, pwd]):
        connection_string = (
            f"DRIVER={{{driver}}};"
            f"HOSTNAME={host};"
            f"PORT={port};"
            f"DATABASE={database};"
            f"UID={uid};"
            f"PWD={pwd};"
            f"PROTOCOL=TCPIP;"
            f"CurrentSchema=MAGICADM;"
        )
    else:
        raise ValueError("Debes especificar un DSN o los parámetros individuales de conexión.")
    
    try:
        with pyodbc.connect(connection_string) as conn:
            # Establecer el schema por defecto
            df = pd.read_sql(sql, conn)
            print(f"✅ Consulta ejecutada correctamente. Registros cargados: {len(df)}")
    except Exception as e:
        print(f"❌ Error al ejecutar la consulta:\n{e}")
        df = None

    return df

#################################################################################################################
#### Función que combina los dataframes necesarios en uno solo
def combine_dfs( mc, fik, doters ):
    ########################################################################
    # Groupby de los valores de FIK
    # Diccionario de agregaciones
    agg_dict = {
        "CAPACIDAD_ASIENTOS_TRAMO": "max",
        "KMS_TRAMO": "max",
        "EMPRESA": lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
        "TIPO_CORRIDA": lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
        "TIPO_BUS": lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None
    }

    # Detectar las columnas restantes para sumarlas
    sum_cols = [c for c in fik.columns if c not in [ "CV_CORRIDA", "FECHA_CORRIDA", "HORA_SALIDA_CORRIDA", "ORIGEN", "DESTINO", "CV_ASIGN", "ORIGEN_CORRIDA", "DESTINO_CORRIDA" ] + list(agg_dict.keys())]

    for col in sum_cols:
        agg_dict[col] = "sum"

    # GroupBy con agregaciones
    fik_grouped = (
        fik
        .groupby([ "CV_CORRIDA", "FECHA_CORRIDA", "HORA_SALIDA_CORRIDA", "ORIGEN", "DESTINO", "CV_ASIGN", "ORIGEN_CORRIDA", "DESTINO_CORRIDA" ], as_index=False)
        .agg(agg_dict)
    )
    del( [agg_dict, sum_cols, col] )

    # Hacemos merge tipo LEFT
    df_prev = pd.merge(
        mc,
        fik_grouped,
        on=[ "CV_CORRIDA", "FECHA_CORRIDA", "HORA_SALIDA_CORRIDA", "ORIGEN", "DESTINO" ]
    )
    #del( [mc, fik, fik_grouped] )

    # Hacemos merge tipo LEFT
    df_final = pd.merge(
        df_prev,
        doters,
        on = [ "TRANSACCION", "FECHA_OPERACION", "FECHA_CORRIDA", "ORIGEN", "DESTINO" ], 
        how = "left"
    )
    #print( df_final )

    # --- Se obtiene una sola columna para identificar método de pago
    df_final["PAGO_METODO"] = df_final.apply(
        lambda row: "TARJETA" if pd.notnull(row["TARJETA"]) else "EFECTIVO",
        axis=1
        )

    # --- Conversión de fechas y horas ---
    df_final["FECHA_OPERACION"] = pd.to_datetime( df_final["FECHA_OPERACION"], errors="coerce" )
    df_final["FECHA_CORRIDA"] = pd.to_datetime( df_final["FECHA_CORRIDA"], errors="coerce" )
    
    # Convertir hora a formato decimal (HH.MM)
    df_final['HORA_OPERACION'] = pd.to_datetime( df_final['HORA_OPERACION'], format='%H:%M:%S' )
    df_final['HORA_SALIDA_CORRIDA'] = pd.to_datetime( df_final['HORA_SALIDA_CORRIDA'], format='%H:%M:%S' )
    df_final['HORA_SALIDA_CORRIDA_'] = df_final['HORA_SALIDA_ORIGEN_CORRIDA'].str[:8]
    df_final['HORA_SALIDA_CORRIDA_'] = pd.to_datetime(df_final['HORA_SALIDA_CORRIDA_'], format='%H:%M:%S' ).dt.time
    # Crear columna con formato NumeroDia_NombreDia
    # 1. Extraer número de día (lunes=0, domingo=6)
    dias_es = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']

    df_final['NUM_DIA'] = df_final['FECHA_OPERACION'].dt.dayofweek
    df_final['NOMBRE_DIA_OPERACION'] = (
        df_final['NUM_DIA'].astype(str) + '_' +  # Número de día (0=Lunes)
        df_final['NUM_DIA'].apply(lambda x: dias_es[x])
    )

    df_final['NUM_DIA'] = df_final['FECHA_CORRIDA'].dt.dayofweek
    df_final['NOMBRE_DIA_CORRIDA'] = (
        df_final['NUM_DIA'].astype(str) + '_' +  # Número de día (0=Lunes)
        df_final['NUM_DIA'].apply(lambda x: dias_es[x])
    )

    df_final.drop(columns=['NUM_DIA'], inplace=True)

    df_final = df_final[df_final["NUM_ASIENTO"].astype(int) <= np.max(df_final['CAPACIDAD_ASIENTOS_TRAMO'])].copy().reset_index(drop=True)
    df_final.loc[df_final["NUM_ASIENTO"].astype(int) > df_final["CAPACIDAD_ASIENTOS_TRAMO"].astype(int), "CAPACIDAD_ASIENTOS_TRAMO"] = np.max(df_final['CAPACIDAD_ASIENTOS_TRAMO'])
    
    return df_final

#################################################################################################################
#### función que optimiza el tipo de datos para disminuir memoria y optimizar cálculos
def optimize_dataframe(df, verbose=True):
    """
    Optimiza el uso de memoria de un DataFrame de Pandas.
    Convierte columnas a tipos más eficientes (int, float, category) cuando es posible.
    
    Parámetros:
        df (pd.DataFrame): DataFrame a optimizar.
        verbose (bool): Si True, imprime el uso de memoria antes y después.

    Retorna:
        pd.DataFrame optimizado.
    """

    # Memoria antes
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memoria usada antes: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            # Optimizar enteros
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            # Optimizar flotantes
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')

        elif pd.api.types.is_datetime64_any_dtype(col_type):
            # Ya es datetime, no tocar
            continue

        else:
            # Intentar convertir fechas en formato string
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
                continue
            except:
                pass

            # Si es texto y hay repetición, usar category
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')

    # Memoria después
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memoria usada después: {end_mem:.2f} MB")
        print(f"Reducción: {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df

#################################################################################################################
#### función que realizar la preparación de los datos para los futuros cálculos
def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    # Cálculo de valores
    df['TOTAL_BOLETOS'] = df.BOLETOS_VEND - df.BOLETOS_CANCEL
    df['TOTAL_VENTA'] = df.VENTA - df.VENTA_CANCEL - df.IVA_VENDIDO - df.IVA_CANCEL
    df['DIF_TARIF'] = np.round( df.TOTAL_VENTA / df.TARIFA_BASE_TRAMO, 4 )

    # Convertir hora a formato decimal (HH.MM)
    #df["HORA_DECIMAL"] = np.round( df["HORA_SALIDA_CORRIDA"].dt.hour + df["HORA_SALIDA_CORRIDA"].dt.minute / 60, 2 )
    df["HORA_DECIMAL"] = df["HORA_SALIDA_CORRIDA"].dt.hour
    
    # --- Agregar columnas útiles ---
    df["AÑO"] = df["FECHA_CORRIDA"].dt.year
    df["MES"] = df["FECHA_CORRIDA"].dt.month

    df["DIAS_ANTICIPACION"] = (df["FECHA_CORRIDA"] - df["FECHA_OPERACION"]).dt.days
    # Unir fecha y hora en una sola columna datetime
    DATETIME_CORRIDA = df["FECHA_CORRIDA"] + pd.to_timedelta(df["HORA_SALIDA_CORRIDA"].dt.hour, unit="h")
    DATETIME_OPERACION = df["FECHA_OPERACION"] + pd.to_timedelta(df["HORA_OPERACION"].dt.hour, unit="h")

    # Diferencia en horas
    df["HORAS_ANTICIPACION"] = ( DATETIME_CORRIDA - DATETIME_OPERACION ).dt.total_seconds() / 3600
    df['TIEMPO_ANTICIPACION'] = np.round( df["HORAS_ANTICIPACION"] / 24, 5 )

    # Cambio de tipo de variabe
    df["CAPACIDAD_ASIENTOS_TRAMO"] = df["CAPACIDAD_ASIENTOS_TRAMO"].astype(int)
    df["NUM_ASIENTO"] = df["NUM_ASIENTO"].astype(int)
    
    return df

