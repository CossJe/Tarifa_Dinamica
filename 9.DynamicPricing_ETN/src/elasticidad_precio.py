import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def _prep_base(df):
    """Prepara columnas básicas, filtra registros inválidos y crea precio promedio."""
    df = df.copy()

    df_resum = df.groupby(
        ['FECHA_OPERACION', 'AÑO', 'NOMBRE_DIA_CORRIDA'
        ]).agg(
            Num_Opers = ('OPERACION', 'count'),
            Tot_Boletos = ('TOTAL_BOLETOS', 'sum'),
            Tot_Venta = ('TOTAL_VENTA', 'sum'),
            Prom_Ocupacion = ('OCUPACION_TRAMO', 'mean'),
            Dias_Anticip = ('DIAS_ANTICIPACION', 'mean'),
            Tarif_Base_Tramo = ('TARIFA_BASE_TRAMO', 'mean'),
            Dif_Tarifa = ('DIF_TARIF', 'mean')
        ).reset_index()

    # Precio promedio efectivo por observación
    df_resum["Costo_Prom_Boleto"] = df_resum["Tot_Venta"] / df_resum["Tot_Boletos"]

    # Filtros mínimos de calidad
    df_resum = df_resum[
        (df_resum["Tot_Boletos"] > 0) &
        (df_resum["Tot_Venta"] > 0) &
        (df_resum["Costo_Prom_Boleto"].replace([np.inf, -np.inf], np.nan).notna())
    ].copy()

    # Asegurar tipos
    for c in ["AÑO", "Dias_Anticip", "Prom_Ocupacion"]:
        if c in df_resum.columns:
            df_resum[c] = pd.to_numeric(df_resum[c], errors="coerce")

    return df_resum


def _deflator_tarifa_base(df, base_year=2025):
    """Construye deflactor anual usando TARIFA_BASE_TRAMO ponderada por boletos."""
    tmp = (
        df.dropna(subset=["Tarif_Base_Tramo"])
          .groupby("AÑO")
          .apply(lambda g: np.average(g["Tarif_Base_Tramo"], weights=g["Tot_Boletos"]))
          .rename("idx")
          .reset_index()
    )
    base = tmp.loc[tmp["AÑO"] == base_year, "idx"]
    if base.empty:
        raise ValueError(f"No hay datos para el año base {base_year} en Tarif_Base_Tramo.")
    base_idx = float(base.iloc[0])
    tmp["deflator"] = tmp["idx"] / base_idx  # >1 si año está por encima del base
    return tmp[["AÑO", "deflator"]]


def _deflator_interno(df, base_year=2025,
                      anticip_min=7, anticip_max=30,
                      occ_min=30, occ_max=70,
                      requiere_no_promo=False):
    """Deflactor anual desde el precio efectivo en condiciones 'estables'."""
    cond = (
        df["Dias_Anticip"].between(anticip_min, anticip_max, inclusive="both") &
        df["Prom_Ocupacion"].between(occ_min, occ_max, inclusive="both")
    )
    if requiere_no_promo and "PORCENT_PROMO" in df.columns:
        cond &= (df["PORCENT_PROMO"] == 0)

    stab = df.loc[cond].copy()
    if stab.empty:
        raise ValueError("No hay datos suficientes en condiciones 'estables' para construir deflactor interno.")

    tmp = (
        stab.groupby("AÑO")
            .apply(lambda g: np.average(g["Costo_Prom_Boleto"], weights=g["Tot_Boletos"]))
            .rename("idx")
            .reset_index()
    )
    base = tmp.loc[tmp["AÑO"] == base_year, "idx"]
    if base.empty:
        raise ValueError(f"No hay datos para el año base {base_year} en el subconjunto estable.")
    base_idx = float(base.iloc[0])
    tmp["deflator"] = tmp["idx"] / base_idx
    return tmp[["AÑO", "deflator"]]


def _fit_loglog(df, precio_col="PRECIO_AJUST", use_year_fe=False):
    """
    Ajusta ln(Q) ~ ln(P) + controles (+ FE de año opcional).
    Controles: DIAS_ANTICIPACION, OCUPACION_TRAMO, dummies de día y clase si existen.
    """
    work = df.dropna(subset=[precio_col, "Tot_Boletos"]).copy()
    work = work[(work[precio_col] > 0) & (work["Tot_Boletos"] > 0)]

    # Variables log
    work["ln_Q"] = np.log1p(work["Tot_Boletos"])
    work["ln_P"] = np.log1p(work[precio_col])

    # Matriz X con controles numéricos
    X_cols = ["ln_P"]
    if "Dias_Anticip" in work.columns:
        X_cols.append("Dias_Anticip")

    X = work[X_cols]
    
    if "Dias_Anticip" in X.columns:
        X['Dias_Anticip'] = np.log1p(X['Dias_Anticip'])

    # Dummies categóricas útiles (día y clase)
    if "NOMBRE_DIA_CORRIDA" in work.columns:
        X = pd.concat([X, pd.get_dummies(work["NOMBRE_DIA_CORRIDA"], prefix="DIA", drop_first=False).astype(int)], axis=1)
    elif "NOMBRE_DIA_OPERACION" in work.columns:
        X = pd.concat([X, pd.get_dummies(work["NOMBRE_DIA_OPERACION"], prefix="DIA", drop_first=True).astype(int)], axis=1)
    '''
    if "CLASE_SERVICIO" in work.columns:
        X = pd.concat([X, pd.get_dummies(work["CLASE_SERVICIO"], prefix="CLASE", drop_first=True).astype(int)], axis=1)
    '''
    # Efectos fijos por año (absorbe inflación/reajustes)
    if use_year_fe and "AÑO" in work.columns:
        X = pd.concat([X, pd.get_dummies(work["AÑO"].astype(int), prefix="ANIO", drop_first=True).astype(int)], axis=1)

    X = sm.add_constant(X, has_constant="add")
    y = work["ln_Q"]

    # Ajuste OLS con errores robustos (HC1)
    model = sm.OLS(y, X).fit()

    pred = model.predict( X )
    
    fig, axs = plt.subplots(2)
    axs[0].plot(work["ln_Q"])
    axs[0].plot(model.predict(X))
    axs[1].scatter( work["ln_Q"], model.predict(X) )
    plt.show()
    
    return model


def elasticidad_precio(
    df_original,
    metodo_ajuste="year_fe",      # 'year_fe' | 'tarifa_base' | 'interno'
    anio_base=2025,
    requiere_no_promo=False,
    devolver_por_anio=True
    ):
    """
    Calcula elasticidad precio con tu modelo log-log + controles,
    aplicando el ajuste por inflación/aumentos según 'metodo_ajuste'.
    """
    df = _prep_base(df_original)

    if metodo_ajuste == "year_fe":
        # No deflacto. Uso efectos fijos por año.
        df["PRECIO_AJUST"] = df["Costo_Prom_Boleto"].copy()
        model = _fit_loglog(df, precio_col="PRECIO_AJUST", use_year_fe=True)

    elif metodo_ajuste == "tarifa_base":
        defl = _deflator_tarifa_base(df, base_year=anio_base)
        df = df.merge(defl, on="AÑO", how="left")
        if df["deflator"].isna().any():
            raise ValueError("Faltan deflactores para algún año (revisa TARIFA_BASE_TRAMO).")
        df["PRECIO_AJUST"] = df["Costo_Prom_Boleto"] / df["deflator"]
        model = _fit_loglog(df, precio_col="PRECIO_AJUST", use_year_fe=False)

    elif metodo_ajuste == "interno":
        defl = _deflator_interno(df, base_year=anio_base, requiere_no_promo=requiere_no_promo)
        df = df.merge(defl, on="AÑO", how="left")
        if df["deflator"].isna().any():
            raise ValueError("Faltan deflactores para algún año en el índice interno.")
        df["PRECIO_AJUST"] = df["Costo_Prom_Boleto"] / df["deflator"]
        model = _fit_loglog(df, precio_col="PRECIO_AJUST", use_year_fe=False)

    else:
        raise ValueError("metodo_ajuste debe ser 'year_fe', 'tarifa_base' o 'interno'.")

    # Elasticidad global (coeficiente de ln_P)
    beta = model.params["ln_P"]
    resumen = {
        "elasticidad_global": float(beta),
        "intervalo_95%": tuple(model.conf_int().loc["ln_P"].values.tolist()),
        "R2_ajustado": float(model.rsquared_adj),
        "metodo": metodo_ajuste
    }

    resultados = {"modelo_global": model, "resumen_global": resumen}

    # (Opcional) Elasticidad por año para comparar
    if devolver_por_anio and "AÑO" in df.columns:
        por_anio = {}
        for anio, g in df.groupby("AÑO"):
            try:
                m = _fit_loglog(g, precio_col="PRECIO_AJUST", use_year_fe=False)
                por_anio[int(anio)] = {
                    "elasticidad": float(m.params["ln_P"]),
                    "intervalo_95%": tuple(m.conf_int().loc["ln_P"].values.tolist()),
                    "R2_ajustado": float(m.rsquared_adj),
                    "modelo": m
                }
            except Exception:
                # Puede fallar por datos insuficientes al desagregar
                continue
        resultados["por_anio"] = por_anio
        
    return resultados
