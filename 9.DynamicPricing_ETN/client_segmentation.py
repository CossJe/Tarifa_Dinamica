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
    #print( display(data.groupby("DIAS_ANTICIPACION")["BOLETOS_VEND"].sum() ) )

    # --- Log seguro ---
    y = np.log(np.where(data[col2eval].values <= 0, np.nan, data[col2eval].values))

    # --- Función de suavizado robusta ---
    def suavizar(y):
        n = len(y)
        if n < 3:
            return np.exp(y)
        
        wl = min(7, n if n % 2 == 1 else n - 1)
        wl = max(3, wl)
        try:
            y_smooth = savgol_filter(y, window_length=wl, polyorder=2)
        except Exception as e:
            print("Error en suavizado:", e)
            y_smooth = y

        # Ajuste último punto
        if n >= 3:
            ultimos2 = np.mean(y_smooth[-3:-1])
            y_smooth[-1] = (y_smooth[-1] + ultimos2) / 2
        elif n == 2:
            y_smooth[-1] = (y_smooth[-1] + y_smooth[-2]) / 2

        y_smooth = np.nan_to_num(y_smooth, nan=np.nan, posinf=np.nan, neginf=np.nan)
        return np.exp(y_smooth)

    y_smooth = suavizar(y)

    mask = (data['DIAS_ANTICIPACION'] >= 15)
    n = mask.sum()

    data[col2eval + '_SMOOTH'] = data[col2eval].copy()

    if n > 0 and len(y_smooth) >= n:
        data.loc[mask, col2eval + '_SMOOTH'] = np.round( y_smooth[-n:], 0 )
    else:
        data[col2eval + '_SMOOTH'] = y_smooth

    # Diagnóstico
    #print("Resumen de NaN después del suavizado:")
    #print(data[[col2eval, col2eval + '_SMOOTH']].isna().sum())

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
    df["Clasif_venta_anticip"] = df["Clasif_venta_anticip_code"].map(mapa_clasificacion).astype("category")

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
        if pct <= 60:
            return "Alta"
        else:
            return "Baja"
        
    def codigo_clasificar(pct):
        if pct <= 60:
            return 1
        else:
            return 0

    # Aplicamos por cada capacidad de autobús
    df["Clasif_asiento"] = df.groupby("CAPACIDAD_ASIENTOS_TRAMO")["PROP_ACUM_PCT"].transform(
        lambda x: x.apply(clasificar).astype("category")
    )
    df["Clasif_asiento_code"] = df.groupby("CAPACIDAD_ASIENTOS_TRAMO")["PROP_ACUM_PCT"].transform(
        lambda x: x.apply(codigo_clasificar)
    )

    return df


def clasificar_meses(df, col_pax="PAX_SUBEN", anio_col="AÑO", mes_col="MES"):
    """
    Clasifica los meses en 'Alto' (1) o 'Bajo' (0) según su proporción acumulada
    de pasajeros ponderados por año (más peso al más reciente).
    
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
        Mismo dataframe de entrada con columnas `Clasif_mes` (Alto/Bajo)
        y `Clasif_mes_code` (1/0) agregadas.
    """

    # 1️ Calcular pesos por año (más reciente = mayor peso)
    anios = sorted(df[anio_col].unique(), reverse=True)
    pesos = {anio: 1/(i+1) for i, anio in enumerate(anios)}
    total = sum(pesos.values())
    pesos = {k: v/total for k, v in pesos.items()}  # normalizamos

    # 2️ Calcular PAX ponderado por año y mes
    df_pond = (
        df.groupby([anio_col, mes_col])[col_pax].sum().reset_index()
    )
    df_pond["peso"] = df_pond[anio_col].map(pesos)
    df_pond["pond"] = df_pond[col_pax] * df_pond["peso"]

    # 3️ Sumar ponderado por mes
    resumen = (
        df_pond.groupby(mes_col)["pond"].sum().reset_index()
    )

    # 4️ Calcular proporciones y proporción acumulada
    resumen["prop_mes"] = resumen["pond"] / resumen["pond"].sum()
    resumen = resumen.sort_values("prop_mes", ascending=False)
    resumen["prop_acum"] = resumen["prop_mes"].cumsum()

    # 5️ Clasificación por proporción acumulada
    resumen["Clasif_mes"] = resumen["prop_acum"].apply(lambda x: "Alto" if x <= 0.6 else "Bajo")
    resumen["Clasif_mes_code"] = resumen["Clasif_mes"].map({"Alto": 1, "Bajo": 0})

    # 6️ Merge con el dataframe original
    df_result = df.merge(resumen[[mes_col, "Clasif_mes", "Clasif_mes_code"]], on=mes_col, how="left")

    return df_result



def clasificar_dia_hora(df, dia_col="NOMBRE_DIA_CORRIDA", hora_col="HORA_DECIMAL", col2var="BOLETOS_VEND"):
    """
    Clasifica por día y por hora en 'Alto' (1) o 'Bajo' (0) según proporciones acumuladas
    del total de la variable `col2var` (por ejemplo, boletos vendidos).

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las columnas de día, hora y variable base (por ejemplo BOLETOS_VEND).
    dia_col : str
        Nombre de la columna del día (ej. 'NOMBRE_DIA_CORRIDA').
    hora_col : str
        Nombre de la columna de la hora (ej. 'HORA_DECIMAL').
    col2var : str
        Nombre de la columna usada para calcular proporciones (ej. 'BOLETOS_VEND').

    Retorna:
    --------
    df : pd.DataFrame
        Mismo dataframe con nuevas columnas:
        - 'Clasif_dia_venta'
        - 'Clasif_hora_venta'
        - 'Clasif_dia_venta_code'
        - 'Clasif_hora_venta_code'
    """

    # -----------------------------
    # Clasificación por DÍA
    # -----------------------------
    # Agrupamos por día y sumamos la variable base
    resumen_dia = df.groupby(dia_col)[col2var].sum().reset_index(name="total_var")
    resumen_dia["prop"] = resumen_dia["total_var"] / resumen_dia["total_var"].sum()
    resumen_dia = resumen_dia.sort_values("prop", ascending=False)
    resumen_dia["prop_acum"] = resumen_dia["prop"].cumsum()

    # Clasificación Alto/Bajo 60/40
    resumen_dia["Clasif_dia_venta"] = resumen_dia["prop_acum"].apply(lambda x: "Alto" if x <= 0.6 else "Bajo")
    resumen_dia["Clasif_dia_venta_code"] = resumen_dia["Clasif_dia_venta"].map({"Alto": 1, "Bajo": 0})

    # -----------------------------
    # Clasificación por HORA
    # -----------------------------
    resumen_hora = df.groupby(hora_col)[col2var].sum().reset_index(name="total_var")
    resumen_hora["prop"] = resumen_hora["total_var"] / resumen_hora["total_var"].sum()
    resumen_hora = resumen_hora.sort_values("prop", ascending=False)
    resumen_hora["prop_acum"] = resumen_hora["prop"].cumsum()

    resumen_hora["Clasif_hora_venta"] = resumen_hora["prop_acum"].apply(lambda x: "Alto" if x <= 0.7 else "Bajo")
    resumen_hora["Clasif_hora_venta_code"] = resumen_hora["Clasif_hora_venta"].map({"Alto": 1, "Bajo": 0})

    # -----------------------------
    # Merge de clasificaciones al dataframe original
    # -----------------------------
    df = df.merge(
        resumen_dia[[dia_col, "Clasif_dia_venta", "Clasif_dia_venta_code"]],
        on=dia_col, how="left"
    )

    df = df.merge(
        resumen_hora[[hora_col, "Clasif_hora_venta", "Clasif_hora_venta_code"]],
        on=hora_col, how="left"
    )

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
