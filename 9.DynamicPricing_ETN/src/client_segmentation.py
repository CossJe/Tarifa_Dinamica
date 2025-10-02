import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

###########################################################################################3
## --- Función que suaviza por Savizky-Golay 
def smooth_data_dynamic_pricing(df: pd.DataFrame, col2eval) -> pd.DataFrame:
    """
    Segmenta datos de compra anticipada de boletos usando Change Point Detection y GMM.
    Se aplica suavizado a la serie de BOLETOS_VEND con filtro Savitzky-Golay.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas ['DIAS_ANTICIPACION','BOLETOS_VEND','VENTA','COSTO_PROM_BOLETO']
    
    Returns:
        pd.DataFrame: DataFrame con columnas extra ['Boletos_Suavizados','Segmento_CPD','Segmento_GMM']
    """

    # Copia para no modificar original
    data = df[df['DIAS_ANTICIPACION'] >= 0].copy().reset_index(drop=True)
    #data = df.copy().reset_index(drop=True)
    
    # Serie objetivo
    y = np.log( data[ col2eval ].values )

    # ----------- Suavizado ----------
    def suavizar(y):
        n = len(y)
        if n < 3:  # Si hay muy pocos datos
            return np.exp( y )

        try:
            y_smooth = savgol_filter(y, window_length=7 if n >= 7 else 3, polyorder=2)
        except ValueError:
            # Si no alcanza para ventana de 7, usar 3
            y_smooth = savgol_filter(y, window_length=3, polyorder=2)

        # Ajustar último punto: promedio entre valor suavizado y promedio de últimos 2 valores suavizados
        if n >= 3:
            ultimos2 = np.mean(y_smooth[-3:-1])  # penúltimo y antepenúltimo
            y_smooth[-1] = (y_smooth[-1] + ultimos2) / 2
        elif n == 2:
            y_smooth[-1] = (y_smooth[-1] + y_smooth[-2]) / 2
        
        return np.exp( y_smooth )

    y_smooth = suavizar(y)

    mask = ( data['DIAS_ANTICIPACION'] >= 15 )
    n =  len( data.loc[ data['DIAS_ANTICIPACION'] >= 15 , 'DIAS_ANTICIPACION' ] )

    data[ col2eval + '_SMOOTH' ] = data[ col2eval ].copy()
    data.loc[mask, col2eval + '_SMOOTH' ] = y_smooth[ -n: ]

    return data

## --- Clasificador de datos por CPD
def CPD_segmentation( data, y_smooth ):
    # ----------- Change Point Detection -----------
    model = rpt.KernelCPD(kernel="linear").fit(y_smooth.reshape(-1,1))
    # número máximo de cambios = 4 segmentos aprox.
    change_points = model.predict(n_bkps=4)
    
    segmento_cpd = np.zeros(len(y_smooth), dtype=int)
    last = 0
    for i, cp in enumerate(change_points):
        segmento_cpd[last:cp] = i
        last = cp
    data['Segmento_CPD'] = segmento_cpd

    return data

## --- Clasificador de datos por GMM
def GMM_segmentation( data, y_smooth ):
    # ----------- Gaussian Mixture Model -----------
    gmm = GaussianMixture(n_components=3, random_state=42)
    data['Segmento_GMM'] = gmm.fit_predict(y_smooth.reshape(-1,1))

    return data


## --- Segmentador de recta de decaimiento exponencial de la demanada por días de anticipación
def decay_line_segment(df, x_col, y_col, max_segmentos = 5, rmse_ini = 1.25, r2_ini = 0.9875, verbose = False):
    # Filtrar solo datos desde día 0
    df = df[df[x_col] >= 0].copy().reset_index(drop=True)
    n = len(df)
    
    segmentos = []
    y_pred_total = np.full(n, np.nan)
    seg_labels = np.full(n, np.nan)

    start_idx = 0
    seg_id = 1

    while start_idx < n and seg_id <= max_segmentos:
        end_idx = n  # probar hasta el final
        mejor_end = None
        mejor_modelo = None
        mejor_pred = None

        if seg_id < max_segmentos:
            while end_idx - start_idx >= (3 if seg_id <= 2 else 7):
                X = df.loc[start_idx:end_idx-1, [x_col]].values
                y = np.log( df.loc[start_idx:end_idx-1, y_col].values )

                modelo = LinearRegression().fit(X, y)
                y_pred = modelo.predict(X)

                rmse = np.sqrt(mean_squared_error(y, y_pred))
                rmse_pct = rmse / np.mean(y) * 100
                r2 = r2_score(y, y_pred)
                
                if seg_id < 3:
                    er_up = 0.250
                    r2_up = 0.025
                else:
                    er_up = 1.0
                    r2_up = 0.1

                if rmse_pct <= (rmse_ini + (seg_id - 1)*er_up) and r2 >= (r2_ini - (seg_id - 1)*r2_up) :
                    mejor_end = end_idx
                    mejor_modelo = modelo
                    mejor_pred = y_pred
                    break  # cumplió, no necesitamos reducir más
                else:
                    end_idx -= 1  # reducir el tramo quitando el último punto

        elif seg_id == max_segmentos:
            X = df.loc[start_idx:n-1, [x_col]].values
            y = np.log( df.loc[start_idx:n-1, y_col].values )

            modelo = LinearRegression().fit(X, y)
            y_pred = modelo.predict(X)

            rmse = np.sqrt(mean_squared_error(y, y_pred))
            rmse_pct = rmse / np.mean(y) * 100
            r2 = r2_score(y, y_pred)

            mejor_end = end_idx
            mejor_modelo = modelo
            mejor_pred = y_pred

        # Si encontramos un segmento válido
        if verbose == True:
            print( f"Número de Curva = {seg_id}\nRMSE = {rmse_pct}\nR2 = {r2}\n=================================" )
            plt.plot( df[x_col], np.log( df[y_col] ), '.-k' )
            plt.plot( X, y_pred, 'r' )
            plt.show()
        if mejor_end:
            y_pred_total[start_idx:mejor_end] = mejor_pred
            seg_labels[start_idx:mejor_end] = seg_id
            segmentos.append((seg_id, start_idx, mejor_end))
            start_idx = mejor_end  # siguiente tramo comienza aquí
            seg_id += 1
        else:
            break  # no se pudo ajustar más rectas

    df["y_pred"] = y_pred_total
    df["Clasif_venta_anticip_code"] = seg_labels#.astype("Int64")

    # Diccionario de mapeo
    mapa_clasificacion = {
        1: "Muy Alto",
        2: "Alto",
        3: "Media",
        4: "Bajo"
    }

    # Crear nueva columna con el texto
    df["Clasif_venta_anticip"] = df["Clasif_venta_anticip_code"].map(mapa_clasificacion)

    return df, segmentos


## --- Segmentador de recta de decaimiento exponencial de la demanada por días de anticipación
def filtro_ponderado(columna: pd.Series) -> pd.Series:
    n = len(columna)
    filtrado = np.zeros(n)

    # primer valor
    filtrado[0] = 0.85*columna.iloc[0] + 0.15*columna.iloc[1]

    # valores intermedios
    for i in range(1, n-1):
        filtrado[i] = (0.15*columna.iloc[i-1] +
                       0.70*columna.iloc[i] +
                       0.15*columna.iloc[i+1])

    # último valor
    filtrado[-1] = 0.85*columna.iloc[-1] + 0.15*columna.iloc[-2]

    return pd.Series(filtrado, index=columna.index)


## --- Segmentador de recta de decaimiento exponencial de la demanada por días de anticipación
def clasificar_asientos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica los asientos en Alta, Media y Baja demanda
    según la proporción acumulada de boletos (PROP_ACUM_PCT).

    - Alta: PROP_ACUM_PCT < 30
    - Media: 30 <= PROP_ACUM_PCT < 60
    - Baja:  PROP_ACUM_PCT >= 60

    La clasificación se hace por capacidad del autobús.

    Parámetros:
    -----------
    df : pd.DataFrame
        Debe contener las columnas:
        ['CAPACIDAD_ASIENTOS_TRAMO', 'NUM_ASIENTO', 'PROP_ACUM_PCT']

    Retorna:
    --------
    df : pd.DataFrame con nueva columna 'CLASIFICACION_DEMANDA'
    """

    df = df.copy()

    # Clasificación condicional por rangos
    def clasificar(pct):
        if pct < 30:
            return "Alta"
        elif pct < 60:
            return "Media"
        else:
            return "Baja"
        
    def codigo_clasificar(pct):
        if pct < 30:
            return 1
        elif pct < 60:
            return 2
        else:
            return 3

    # Aplicamos por cada capacidad de autobús
    df["Clasif_asiento"] = df.groupby("CAPACIDAD_ASIENTOS_TRAMO")["PROP_ACUM_PCT"].transform(
        lambda x: x.apply(clasificar)
    )
    df["Clasif_asiento_codigo"] = df.groupby("CAPACIDAD_ASIENTOS_TRAMO")["PROP_ACUM_PCT"].transform(
        lambda x: x.apply(codigo_clasificar)
    )

    return df


def clasificar_meses(df, col_pax="PAX_SUBEN", anio_col="AÑO", mes_col="MES"):
    """
    Clasifica los meses en 'Alto', 'Bajo' o 'Normal' según los pasajeros,
    usando un promedio ponderado por año (más peso al más reciente).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Debe contener columnas [AÑO, MES, PAX_SUBEN].
    col_pax : str
        Nombre de la columna con pasajeros subidos.
    anio_col : str
        Nombre de la columna de año.
    mes_col : str
        Nombre de la columna de mes.
    
    Retorna:
    --------
    df_result : pd.DataFrame
        Mismo dataframe de entrada con columna `Clasif_mes` agregada.
    """

    # Obtener años ordenados de más reciente a más antiguo
    anios = sorted(df[anio_col].unique(), reverse=True)
    pesos = {anio: 1/(i+1) for i, anio in enumerate(anios)}  # peso 1, 0.5, 0.33, etc.
    
    # Normalizar los pesos para que sumen 1
    total = sum(pesos.values())
    pesos = {k: v/total for k,v in pesos.items()}
    
    # Calcular valor ponderado por mes
    df_pond = (
        df.groupby([anio_col, mes_col])[col_pax].sum().reset_index()
    )
    df_pond["peso"] = df_pond[anio_col].map(pesos)
    df_pond["pond"] = df_pond[col_pax] * df_pond["peso"]
    
    resumen = (
        df_pond.groupby(mes_col)["pond"].sum().reset_index()
    )
    
    # Ordenar meses por pasajeros ponderados
    resumen = resumen.sort_values("pond", ascending=False)
    resumen["rank"] = resumen["pond"].rank(method="first", ascending=False).astype(int) - 1  # 0–11
    
    # Clasificación: 0–3 = Alto, 4–7 = Normal, 8–11 = Bajo
    def clasif(r):
        if r <= 3:
            return "Alto"
        elif r >= 8:
            return "Bajo"
        else:
            return "Normal"
    
    resumen["Clasif_mes"] = resumen["rank"].apply(clasif)
    
    # Merge con el df original
    df_result = df.merge(resumen[[mes_col, "Clasif_mes"]], on=mes_col, how="left")
    
    return df_result


def clasificar_dia_hora(df, dia_col, hora_col, col2var):
    """
    Clasifica por NOMBRE_DIA_CORRIDA y HORA_DECIMAL según el valor de FOP_PROM.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con el resumen de FOP_PROM por NOMBRE_DIA_CORRIDA y HORA_DECIMAL.
    dia_col : list
        Columnas en común para hacer la clasificacion por dia.
    hora_col : list
        Columnas en común para hacer la clasificacion por hora.
    col2var : list
        Columnas en común en la que se hará la clasificación

    Retorna:
    --------
    df_completo : pd.DataFrame
        DataFrame con dos nuevas columnas de clasificación:
        - Clasif_dia_venta
        - Clasif_hora_venta
    """
    
    # Clasificación por día
    df["Clasif_dia_venta"] = df.groupby( dia_col )[ col2var ]\
        .transform(lambda x: "Alto" if x.mean() >= 0.5 else "Bajo")
    
    # Clasificación por hora
    df["Clasif_hora_venta"] = df.groupby( hora_col )[ col2var ]\
        .transform(lambda x: "Alto" if x.mean() >= 0.5 else "Bajo")
    
    return df



def merge_with_classification(df_base: pd.DataFrame, 
                              df_resumen: pd.DataFrame, 
                              merge_on: list, 
                              cols_merge: list,
                              verbose=False) -> pd.DataFrame:
    """
    Une un dataframe base con un dataframe resumen que contiene clasificaciones u otros KPIs.

    Parámetros
    ----------
    df_base : pd.DataFrame
        DataFrame original con todos los datos.
    df_resumen : pd.DataFrame
        DataFrame que contiene la clasificación o KPIs ya calculados.
    merge_on : list
        Columnas clave para realizar el merge (ejemplo: ['AÑO', 'MES'] o ['MES']).
    cols_merge : list
        Columnas de df_resumen que quieres agregar a df_base.

    Retorna
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas unidas desde df_resumen.
    """
    # Filtrar solo columnas necesarias en el df_resumen
    df_resumen_filtered = df_resumen[merge_on + cols_merge]

    # Asegurar que las columnas de merge tengan el mismo tipo de datos
    for col in merge_on:
        if col in df_base.columns and col in df_resumen_filtered.columns:
            # Obtener el tipo de dato de la columna en df_base
            base_dtype = df_base[col].dtype
            
            # Convertir la columna en df_resumen al mismo tipo que en df_base
            try:
                df_resumen_filtered[col] = df_resumen_filtered[col].astype(base_dtype)
                if verbose==True:
                    print(f"Columna '{col}' convertida a tipo {base_dtype} en df_resumen")
            except Exception as e:
                print(f"Error al convertir columna '{col}' a tipo {base_dtype}: {e}")
                # Si la conversión falla, intentar convertir a string (tipo más flexible)
                try:
                    df_base[col] = df_base[col].astype(str)
                    df_resumen_filtered[col] = df_resumen_filtered[col].astype(str)
                    if verbose==True:
                        print(f"Columnas '{col}' convertidas a string como fallback")
                except Exception as e2:
                    if verbose==True:
                        print(f"Error crítico al convertir columna '{col}': {e2}")
                    raise

    if verbose == True:
        print( df_base[ merge_on ].dtypes )
        print( df_resumen_filtered[ merge_on ].dtypes )

    # Hacer el merge (left join para no perder filas del df_base)
    df_merged = df_base.merge(df_resumen_filtered, on=merge_on, how='left')

    return df_merged
