import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta
import os

from src.dynamic_pricing_data_loader import cargar_y_preparar_datos

def Get_Daily_Resume( df, cols2an, Plotting_Resume=False ):
    df_agg = (
        df.groupby(
            cols2an,
            as_index=False
        )
        .agg({
            'VENTA': 'sum',
            'BOLETOS_VEND': 'sum',
            'PAX_SUBEN' : 'sum',
            'TARIFA_BASE_TRAMO': 'mean',
            'OCUPACION_TRAMO': 'mean',
            'CAPACIDAD_ASIENTOS_TRAMO' : 'median',
            'PORCENT_PROMO' : 'mean'
        })
    )
    df_agg = df_agg.fillna(0)

    df_agg[ 'TARIF_PROM' ] = df_agg[ 'VENTA' ] / df_agg[ 'BOLETOS_VEND' ]

    # Ordenar por fecha
    df_agg.sort_values("FECHA_OPERACION", inplace=True)

    if Plotting_Resume:
        # Crear figura y eje principal
        fig, ax1 = plt.subplots(figsize=(12,6))

        # --- Eje principal: Tarifas ---
        ax1.plot(df_agg['FECHA_OPERACION'], df_agg['TARIF_PROM'], 
                color='red', label='TARIFA_PROMEDIO_DIARIA', alpha=0.75)
        ax1.plot(df_agg['FECHA_OPERACION'], df_agg['TARIFA_BASE_TRAMO'], 
                color='darkblue', label='TARIFA_BASE', alpha=0.65)

        ax1.set_xlabel('FECHA_OPERACION')
        ax1.set_ylabel('TARIFA', color='darkred')
        ax1.tick_params(axis='y', labelcolor='darkred')
        ax1.grid(True, alpha=0.3)

        # --- Eje secundario: Boletos vendidos ---
        ax2 = ax1.twinx()
        ax2.plot(df_agg['FECHA_OPERACION'], df_agg['BOLETOS_VEND'], 
                color='darkgreen', label='BOLETOS_VENDIDOS', alpha=0.75)
        ax2.set_ylabel('BOLETOS VENDIDOS', color='darkgreen')
        ax2.tick_params(axis='y', labelcolor='darkgreen')

        # --- Título y leyenda combinada ---
        fig.suptitle('Elasticidad Precio-Demanda Diaria (controlada por hora, anticipación, asiento, mes y día)', fontsize=12)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.show()

    return df_agg


# ============================
# FUNCIONES NUEVAS
# ============================
def Calcular_Elasticidad_Global(df_agg, plot_elasticidad=False, usar_wls=False, 
                                  controles_adicionales=True):
    """
    Calcula elasticidad precio-demanda con múltiples mejoras avanzadas.
    
    Parámetros:
    -----------
    df_agg : DataFrame
        Debe contener: FECHA_OPERACION, BOLETOS_VEND, VENTA, TARIF_PROM
    plot_elasticidad : bool
        Si True, genera gráficos de diagnóstico
    usar_wls : bool
        Si True, usa Weighted Least Squares (pondera por volumen de ventas)
    controles_adicionales : bool
        Si True, incluye variables de control avanzadas
    
    Retorna:
    --------
    dict con: elasticidad, intervalo_confianza, r2, p_valor, diagnósticos
    """
    df = df_agg.sort_values('TARIF_PROM').copy()
    
    # 1. Preparación y limpieza
    df = df[(df['TARIF_PROM'] > 0) & (df['BOLETOS_VEND'] > 0)].reset_index(drop=True)
    df['FECHA_OPERACION'] = pd.to_datetime(df['FECHA_OPERACION'])
    df = df.sort_values('FECHA_OPERACION')
    
    # 2. Variables transformadas
    df['log_P'] = np.log(df['TARIF_PROM'])
    df['log_Q'] = np.log(df['BOLETOS_VEND'])
    
    # 3. Variables de control temporal BÁSICAS
    df['dia_semana'] = df['FECHA_OPERACION'].dt.dayofweek
    df['mes'] = df['FECHA_OPERACION'].dt.month
    df['tendencia'] = np.arange(len(df))
    
    # 4. MEJORA 1: Variables de control AVANZADAS
    if controles_adicionales:
        # Fin de semana (afecta demanda)
        df['fin_semana'] = (df['dia_semana'] >= 5).astype(int)
        
        # Temporada alta (ejemplo: vacaciones - ajusta según tu caso)
        df['temp_alta'] = df['mes'].isin([6, 7, 8, 12]).astype(int)
        
        # Días festivos aproximados (ejemplo México - ajusta a tu país)
        df['dia_festivo'] = (
            ((df['FECHA_OPERACION'].dt.month == 1) & (df['FECHA_OPERACION'].dt.day == 1)) |  # Año nuevo
            ((df['FECHA_OPERACION'].dt.month == 12) & (df['FECHA_OPERACION'].dt.day == 25))   # Navidad
        ).astype(int)
        
        # Rezago de demanda (autocorrelación)
        df['log_Q_lag1'] = df['log_Q'].shift(1)
        df['log_Q_lag7'] = df['log_Q'].shift(7)  # Efecto semanal
        
        # Variación de precio (captura ajustes bruscos)
        df['delta_precio'] = df['log_P'].diff().fillna(0)
        
        # Índice de concentración temporal (detecta picos)
        df['rolling_std'] = df['BOLETOS_VEND'].rolling(7, min_periods=1).std()
    
    # Eliminar NaN de los rezagos
    df = df.dropna().reset_index(drop=True)
    
    # 5. One-hot encoding para estacionalidad
    df_dummies = pd.get_dummies(df[['dia_semana', 'mes']], 
                                  columns=['dia_semana', 'mes'], 
                                  drop_first=True)
    
    # 6. MEJORA 2: Construcción de matriz X con controles seleccionados
    X_cols = ['log_P', 'tendencia']
    
    if controles_adicionales:
        X_cols.extend(['fin_semana', 'temp_alta', 'dia_festivo', 
                      'log_Q_lag1', 'log_Q_lag7', 'delta_precio'])
    
    X = pd.concat([
        df[X_cols],
        df_dummies
    ], axis=1).astype(float)
    
    y = df['log_Q'].values
    
    # 7. MEJORA 3: Weighted Least Squares (WLS)
    # Pondera por volumen de ventas (observaciones con más boletos son más confiables)
    if usar_wls:
        pesos = np.sqrt(df['BOLETOS_VEND'].values)  # Raíz cuadrada para estabilidad
        pesos = pesos / pesos.mean()  # Normalizar
        
        # Aplicar pesos multiplicando X e y
        X_weighted = X.values * pesos.reshape(-1, 1)
        y_weighted = y * pesos
        
        model = LinearRegression().fit(X_weighted, y_weighted)
        y_pred = model.predict(X.values)  # Predicción sin pesos para residuos
    else:
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        pesos = np.ones(len(df))
    
    residuos = y - y_pred
    
    # 8. Elasticidad e inferencia estadística
    elasticidad = float(model.coef_[0])
    
    # Error estándar robusto (ajustado por heterocedasticidad)
    n = len(df)
    k = X.shape[1]
    
    # Errores estándar robustos de White
    X_array = X.values
    residuos_sq = residuos ** 2
    
    # Matriz de covarianza robusta
    XtX_inv = np.linalg.inv(X_array.T @ X_array)
    meat = X_array.T @ np.diag(residuos_sq * pesos**2) @ X_array  # Ajuste por pesos
    var_covar_robust = XtX_inv @ meat @ XtX_inv
    se_elasticidad = np.sqrt(var_covar_robust[0, 0])
    
    # Intervalo de confianza 95%
    t_crit = stats.t.ppf(0.975, n - k)
    ic_inferior = elasticidad - t_crit * se_elasticidad
    ic_superior = elasticidad + t_crit * se_elasticidad
    
    # P-valor
    t_stat = elasticidad / se_elasticidad
    p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
    
    # R² ajustado
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    # 9. MEJORA 4: Diagnósticos avanzados
    diagnosticos = {
        'durbin_watson': durbin_watson(residuos),
        'jarque_bera': stats.jarque_bera(residuos),
        'breusch_pagan': breusch_pagan_test(residuos, X_array),
        'n_observaciones': n,
        'vif_precio': calcular_vif(X, 0) if X.shape[1] > 1 else None,
        'peso_promedio': pesos.mean(),
        'peso_std': pesos.std()
    }
    
    # 10. Visualizaciones mejoradas
    if plot_elasticidad:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Gráfico 1: Relación log-log con tamaño por volumen
        ax1 = fig.add_subplot(gs[0, :2])
        scatter = ax1.scatter(df['log_P'], df['log_Q'], 
                             s=pesos*50, alpha=0.5, c=df['BOLETOS_VEND'], 
                             cmap='viridis')
        
        # Línea de regresión
        x_range = np.linspace(df['log_P'].min(), df['log_P'].max(), 100)
        #y_range = model.coef_[0] * x_range + model.intercept_ 
        # Intercepto ajustado: considera el valor promedio de TODOS los controles
        intercepto_ajustado = df['log_Q'].mean() - model.coef_[0] * df['log_P'].mean()
        y_range = model.coef_[0] * x_range + intercepto_ajustado
        ax1.plot(x_range, y_range, 'r-', linewidth=2, 
                label=f'Elasticidad: {elasticidad:.3f}')
        #ax1.plot(X['log_P'], y_pred, 'r-', linewidth=2, 
        #        label=f'Elasticidad: {elasticidad:.3f}')
        
        
        ax1.set_xlabel('log(Tarifa Promedio)', fontsize=11)
        ax1.set_ylabel('log(Boletos Vendidos)', fontsize=11)
        ax1.set_title(f'Elasticidad Global: {elasticidad:.3f} IC95%: [{ic_inferior:.3f}, {ic_superior:.3f}]', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Boletos Vendidos')
        
        # Gráfico 2: Importancia de variables
        ax2 = fig.add_subplot(gs[0, 2])
        coef_importancia = pd.DataFrame({
            'Variable': X.columns[:8],  # Top 8
            'Coeficiente': np.abs(model.coef_[:8])
        }).sort_values('Coeficiente', ascending=True)
        
        ax2.barh(coef_importancia['Variable'], coef_importancia['Coeficiente'])
        ax2.set_xlabel('|Coeficiente|')
        ax2.set_title('Importancia de Variables')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Gráfico 3: Residuos vs ajustados
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(y_pred, residuos, alpha=0.5, s=30)
        ax3.axhline(0, color='r', linestyle='--')
        ax3.set_xlabel('Valores Ajustados')
        ax3.set_ylabel('Residuos')
        ax3.set_title('Residuos vs Ajustados')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Q-Q plot
        ax4 = fig.add_subplot(gs[1, 1])
        stats.probplot(residuos, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Distribución de residuos
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(residuos, bins=30, edgecolor='black', alpha=0.7)
        ax5.axvline(0, color='r', linestyle='--')
        ax5.set_xlabel('Residuos')
        ax5.set_ylabel('Frecuencia')
        ax5.set_title('Distribución de Residuos')
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Serie temporal de residuos
        ax6 = fig.add_subplot(gs[2, :])
        ax6.plot(df['FECHA_OPERACION'], residuos, alpha=0.7, linewidth=1)
        ax6.axhline(0, color='r', linestyle='--')
        ax6.fill_between(df['FECHA_OPERACION'], residuos, 0, alpha=0.3)
        ax6.set_xlabel('Fecha')
        ax6.set_ylabel('Residuos')
        ax6.set_title(f'Residuos Temporales (Durbin-Watson: {diagnosticos["durbin_watson"]:.3f})')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 11. Resultado completo
    return {
        'elasticidad': elasticidad,
        'intervalo_confianza_95': (ic_inferior, ic_superior),
        'error_estandar': se_elasticidad,
        'p_valor': p_valor,
        'r2': r2,
        'r2_ajustado': r2_ajustado,
        'significativo': p_valor < 0.05,
        'diagnosticos': diagnosticos,
        'modelo': model,
        'coeficientes': dict(zip(X.columns, model.coef_)),
        'interpretacion': interpretar_elasticidad(elasticidad, p_valor, diagnosticos)
    }

def durbin_watson(residuos):
    """Calcula estadístico Durbin-Watson para autocorrelación"""
    diff = np.diff(residuos)
    return np.sum(diff**2) / np.sum(residuos**2)

def breusch_pagan_test(residuos, X):
    """Test de Breusch-Pagan para heterocedasticidad"""
    n = len(residuos)
    residuos_sq = residuos ** 2
    
    # Regresión auxiliar
    aux_model = LinearRegression().fit(X, residuos_sq)
    ss_explained = np.sum((aux_model.predict(X) - residuos_sq.mean()) ** 2)
    ss_total = np.sum((residuos_sq - residuos_sq.mean()) ** 2)
    
    r2_aux = ss_explained / ss_total if ss_total > 0 else 0
    lm_stat = n * r2_aux
    p_valor = 1 - stats.chi2.cdf(lm_stat, X.shape[1])
    
    return {'estadistico': lm_stat, 'p_valor': p_valor}

def calcular_vif(X, col_index):
    """Calcula Variance Inflation Factor para multicolinealidad"""
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    y = X[:, col_index]
    X_otros = np.delete(X, col_index, axis=1)
    
    model = LinearRegression().fit(X_otros, y)
    r2 = model.score(X_otros, y)
    
    vif = 1 / (1 - r2) if r2 < 0.99 else np.inf
    return vif

def interpretar_elasticidad(e, p_valor, diag):
    """Genera interpretación textual completa"""
    if p_valor >= 0.05:
        return "⚠️ Elasticidad NO significativa estadísticamente (p > 0.05)"
    
    tipo = "elástica" if abs(e) > 1 else "inelástica"
    magnitud = abs(e)
    
    interpretacion = f"✓ La demanda es {tipo} (|E| = {magnitud:.3f}). "
    interpretacion += f"Un aumento del 1% en precio genera una disminución del {magnitud:.2%} en boletos vendidos.\n\n"
    
    # Advertencias de diagnóstico
    dw = diag['durbin_watson']
    if dw < 1.5 or dw > 2.5:
        interpretacion += f"⚠️ Autocorrelación detectada (DW={dw:.2f}). Considerar más rezagos.\n"
    
    if diag['breusch_pagan']['p_valor'] < 0.05:
        interpretacion += "⚠️ Heterocedasticidad presente. Errores estándar robustos aplicados.\n"
    
    if diag.get('vif_precio') and diag['vif_precio'] > 10:
        interpretacion += f"⚠️ Multicolinealidad alta (VIF={diag['vif_precio']:.1f}). Revisar controles.\n"
    
    return interpretacion


def Get_Resampled_Tarify( df_agg, Resamp_Interval=10, Plotting_Resamp=False ):
    # --- 1️⃣ Definir intervalos de 5 pesos ---
    tarif_min = np.floor(df_agg['TARIF_PROM'].min() / Resamp_Interval) * Resamp_Interval
    tarif_max = np.ceil(df_agg['TARIF_PROM'].max() / Resamp_Interval) * Resamp_Interval
    intervalos = np.arange(tarif_min, tarif_max + Resamp_Interval, Resamp_Interval)

    # --- 2️⃣ Asignar cada valor a su intervalo ---
    df_agg['TARIF_BIN'] = pd.cut(df_agg['TARIF_PROM'], bins=intervalos, right=False)

    # --- 3️⃣ Agregar el valor central del bin ---
    df_agg['TARIF_PROM_BIN'] = df_agg['TARIF_PROM'].copy()
    df_agg['TARIF_PROM'] = df_agg['TARIF_BIN'].apply(lambda x: x.left + Resamp_Interval/2).astype( float )

    # --- 4️⃣ Calcular promedios por intervalo ---
    df_resampled = (
        df_agg.groupby('TARIF_PROM')
        .agg({
            'VENTA': 'mean',
            'BOLETOS_VEND': 'mean',
            'PAX_SUBEN': 'mean',
            'TARIFA_BASE_TRAMO': 'mean',
            'OCUPACION_TRAMO': 'mean',
            'CAPACIDAD_ASIENTOS_TRAMO': 'mean',
            'PORCENT_PROMO': 'mean',
            'TARIF_PROM_BIN': 'mean'
        })
        .reset_index()
    )

    # --- 5️⃣ Interpolación inversa a la distancia (IDW) ---
    def idw_interpolation(x_known, y_known, x_missing, power=2):
        """
        Interpola por inverso de la distancia (IDW)
        x_known: posiciones conocidas (1D)
        y_known: valores conocidos (1D)
        x_missing: posiciones donde interpolar
        """
        x_known = np.array(x_known)
        y_known = np.array(y_known)
        results = []
        for xm in x_missing:
            dists = np.abs(x_known - xm)
            # evitar división entre cero
            dists[dists == 0] = 1e-6
            weights = 1 / (dists ** power)
            val = np.sum(weights * y_known) / np.sum(weights)
            results.append(val)
        return np.array(results)

    # --- 6️⃣ Aplicar IDW a todas las variables ---
    numeric_cols = ['VENTA','BOLETOS_VEND','PAX_SUBEN','TARIFA_BASE_TRAMO',
                    'OCUPACION_TRAMO','CAPACIDAD_ASIENTOS_TRAMO','PORCENT_PROMO','TARIF_PROM_BIN']

    for col in numeric_cols:
        mask_nan = df_resampled[col].isna()
        if mask_nan.any():
            x_known = df_resampled.loc[~mask_nan, 'TARIF_PROM']
            y_known = df_resampled.loc[~mask_nan, col]
            x_missing = df_resampled.loc[mask_nan, 'TARIF_PROM']
            df_resampled.loc[mask_nan, col] = idw_interpolation(x_known, y_known, x_missing)

    # --- 7️⃣ Resultado final ---
    df_resampled = df_resampled.sort_values('TARIF_PROM').reset_index(drop=True)

    if Plotting_Resamp:
        plt.figure(figsize=(12,6))
        plt.plot(df_resampled['TARIF_PROM'], df_resampled['BOLETOS_VEND'], color='darkblue', label='BOLETOS VENDIDOS', alpha=0.75)
        plt.axvline(x=df_agg['TARIFA_BASE_TRAMO'].mean(), label=f"Tarifa_Base = {np.round( df_agg['TARIFA_BASE_TRAMO'].mean(), 0)}", color='red', linestyle='--', alpha=0.65)
        plt.axvline(x=df_agg[ 'TARIF_PROM' ].mean(), label=f"Costo_Prom = {np.round( df_agg[ 'TARIF_PROM' ].mean(), 0)}", color='orange', linestyle='--', alpha=0.65)
        plt.legend()
        plt.title('Elasticidad Precio-Demanda Diaria (controlada por hora, anticipación, asiento, mes y día)')
        plt.xlabel('TARIF_PROM')
        plt.ylabel('BOLETOS_VEND')
        plt.grid(True)
        plt.show()

    return df_resampled



def suavizar_con_bordes(df, col, window=7):
    """
    Aplica suavizado por mediana móvil y luego media móvil,
    reemplazando los bordes con los valores originales después de cada paso.
    
    Parámetros:
    df : DataFrame
    col : str, nombre de la columna a suavizar
    window : int, tamaño de ventana (por defecto 7)
    
    Retorna:
    Serie suavizada con bordes conservados.
    """
    series = df[col].copy()
    half = window // 2

    # Paso 1: mediana móvil
    mediana = series.rolling(window, center=True, min_periods=1).median()
    # Rellenar bordes con los valores originales
    mediana.iloc[:half] = series.iloc[:half]
    mediana.iloc[-half:] = series.iloc[-half:]

    # Paso 2: media móvil sobre la serie suavizada
    media = mediana.rolling(window, center=True, min_periods=1).mean()
    # Rellenar bordes nuevamente con los valores originales
    media.iloc[:half] = series.iloc[:half]
    media.iloc[-half:] = series.iloc[-half:]

    return media


def ajustar_tendencia_polinomica(df, x_col, y_col, grados=(1, 2, 3)):
    """
    Ajusta una regresión polinómica entre dos columnas y elige automáticamente el mejor grado (según R²).

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos.
    x_col : str
        Nombre de la columna independiente (X).
    y_col : str
        Nombre de la columna dependiente (Y).
    grados : iterable (por defecto = (2, 3, 4))
        Grados de polinomio a probar.

    Retorna:
    --------
    df_out : DataFrame con columnas:
        - y_tendencia : valores ajustados del mejor modelo
        - y_residuo : residuales
    model : modelo LinearRegression entrenado
    best_degree : grado polinómico seleccionado
    r2_best : coeficiente de determinación del mejor modelo
    """

    X = np.array(df[x_col]).reshape(-1, 1)
    y = np.array(df[y_col])

    best_model = None
    best_r2 = -np.inf
    best_degree = None
    best_pred = None

    for d in grados:
        poly = PolynomialFeatures(degree=d)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_degree = d
            best_model = model
            best_pred = y_pred
            best_poly = poly

    # Guardar resultados
    df_out = df.copy()
    df_out[f'{y_col}_Tendencia'] = best_pred
    df_out[f'{y_col}_Residuo'] = y - best_pred

    print(f"✅ Mejor grado polinómico: {best_degree} | R² = {best_r2:.4f}")

    return df_out, best_model, best_poly, best_degree, best_r2



def encontrar_tarifa_optima(model, poly, p_min=None, p_max=None, n_points=500, c=None):
    """
    Calcula la tarifa (precio) óptima que maximiza ingresos o beneficios 
    a partir de un modelo polinómico ya entrenado.

    Parámetros
    ----------
    model : sklearn.linear_model.LinearRegression
        Modelo ajustado con PolynomialFeatures.
    poly : sklearn.preprocessing.PolynomialFeatures
        Transformador usado para crear X_poly.
    p_min, p_max : float
        Rango de precios a evaluar (si no se da, se infiere del modelo).
    n_points : int
        Resolución del barrido de precios (más alto = más preciso).
    c : float, opcional
        Costo marginal por pasajero. 
        Si se proporciona, se optimiza (P - c)*Q(P) en vez de P*Q(P).

    Retorna
    -------
    dict con:
        - 'P_optimo': tarifa óptima
        - 'Q_optimo': cantidad estimada
        - 'Ingreso_optimo': ingreso (o beneficio) máximo
        - 'rangos': (P_min, P_max) sugeridos +/-10% alrededor del óptimo
        - 'P_grid', 'R_grid', 'Q_grid' para graficar si se desea
    """
    # Inferir rango de precios si no se da
    if p_min is None or p_max is None:
        raise ValueError("Debes definir p_min y p_max (rango de precios a evaluar)")

    # Grid de precios
    P_grid = np.linspace(p_min, p_max, n_points).reshape(-1, 1)
    X_poly = poly.transform(P_grid)
    Q_pred = model.predict(X_poly)

    # Eliminar valores negativos de demanda predicha
    Q_pred = np.where(Q_pred > 0, Q_pred, 0)

    if c is None:
        R_grid = P_grid.flatten() * Q_pred  # ingreso
    else:
        R_grid = (P_grid.flatten() - c) * Q_pred  # beneficio neto

    idx_opt = np.argmax(R_grid)
    P_optimo = float(P_grid[idx_opt])
    Q_optimo = float(Q_pred[idx_opt])
    R_optimo = float(R_grid[idx_opt])

    # Rangos +/-10% en torno al óptimo, recortados al rango original
    rango_min = max(p_min, P_optimo * 0.9)
    rango_max = min(p_max, P_optimo * 1.1)

    resultados = {
        'P_optimo': P_optimo,
        'Q_optimo': Q_optimo,
        'Ingreso_optimo': R_optimo,
        'rangos': (rango_min, rango_max),
        'P_grid': P_grid.flatten(),
        'Q_grid': Q_pred,
        'R_grid': R_grid
    }

    # Exportar datos a json
    resultados_json = {
        'P_optimo': float(P_optimo),
        'Q_optimo': float(Q_optimo),
        'Ingreso_optimo': float(R_optimo),
        'P_rangos': [float(rango_min), float(rango_max)]
    }

    return resultados, resultados_json



def Calculate_Min_Max_Optimal_Tarify( df, days_before, grados=(1, 2, 3), resampling=True, resamp_interval = 10, window=7,
                                    Plotting_Resume=False, Plotting_Resamp=False, Plotting_Results=False, Plotting_Elasticity=False,
                                    path_fold=None ):
    
    # Copia de seguridad
    df['FECHA_OPERACION'] = pd.to_datetime(df['FECHA_OPERACION'])
    df = df[ df["TIPO_PASAJERO"] == "AD" ].copy()
    df = df[ df["VENTA"] > 0 ]

    if isinstance(days_before, (int, float)):
        # Calculate yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        # Calculate the start date (366 days before yesterday to get a full 365 days of data)
        start_date = yesterday - timedelta(days=days_before)

        # Convert to a date-only format to ignore time when filtering
        #yesterday = yesterday.date()
        #start_date = start_date.date()

        df = df.loc[
            (df['FECHA_OPERACION'] >= start_date) & (df['FECHA_OPERACION'] <= yesterday)
        ]
    elif isinstance(days_before, str) and days_before.upper() == "ALL":
        # Calculate yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        df = df.loc[
            df['FECHA_OPERACION'] <= yesterday
        ]

    cols2an = [ 'FECHA_OPERACION' ]
    # Se crea la agrupación de los datos por fecha
    df_agg = Get_Daily_Resume( df, cols2an, Plotting_Resume=Plotting_Resume )

    # Se Remuestrea por rango tarifario para tener mejor apreciación del comportamiento de los datos
    if resampling:
        df_resampled = Get_Resampled_Tarify( df_agg, Resamp_Interval=resamp_interval, Plotting_Resamp=Plotting_Resamp )
    else:
        df_resampled = df_agg

    # --- Calcular elasticidades ---
    #df_elast, E_diaria = Calcular_Elasticidad_Diaria(df_agg, plot_elasticidad=Plotting_Elasticity)
    E_global = Calcular_Elasticidad_Global(df_agg, plot_elasticidad=Plotting_Elasticity)
    print( f"\nInterpretacion de la elasticidad global: \n{E_global['interpretacion']}\n" )

    # --- Cuavizado de los datos
    df_resampled.sort_values("TARIF_PROM", inplace=True)
    df_resampled['TARIF_PROM_Suavizada'] = suavizar_con_bordes(df_resampled, 'TARIF_PROM', window=window)
    df_resampled['BOLETOS_VEND_Suavizada'] = suavizar_con_bordes(df_resampled, 'BOLETOS_VEND', window=window)
    df_resampled['VENTA_Suavizada'] = suavizar_con_bordes(df_resampled, 'VENTA', window=window)

    df_resampled, modelo, poly, grado, r2 = ajustar_tendencia_polinomica(
        df_resampled,
        x_col='TARIF_PROM_Suavizada',
        y_col='VENTA_Suavizada',
        grados = grados
    )

    df_resampled, modelo, poly, grado, r2 = ajustar_tendencia_polinomica(
        df_resampled,
        x_col='TARIF_PROM_Suavizada',
        y_col='BOLETOS_VEND_Suavizada',
        grados = grados
    )

    # Rango realista de tarifas (puedes ajustarlo a tus datos)
    p_min = df_resampled['TARIF_PROM_Suavizada'].min()
    p_max = df_resampled['TARIF_PROM_Suavizada'].max()

    resultado_opt, resultado_json = encontrar_tarifa_optima(
        model=modelo,          # de tu regresión polinómica
        poly=poly,             # de PolynomialFeatures
        p_min=p_min,
        p_max=p_max,
        c=None                 # o c=coste_marginal si quieres maximizar beneficio
    )

    # Imprimir resultados
    print(f"📈 Elasticidad Precio-Demanda: {E_global['elasticidad']:.2f}")
    print(f"💰 Tarifa óptima: {resultado_opt['P_optimo']:.2f}")
    print(f"📈 Cantidad estimada: {resultado_opt['Q_optimo']:.1f}")
    print(f"💵 Ingreso máximo estimado: {resultado_opt['Ingreso_optimo']:.2f}")
    print(f"📊 Rango sugerido: {resultado_opt['rangos'][0]:.2f} – {resultado_opt['rangos'][1]:.2f}")

    # --- Añadir elasticidades al JSON ---
    #df_agg['Elasticidad_Diaria'] = E_diaria
    resultado_json['Elasticidad_Global'] = E_global['elasticidad']

    # Exportar resultados optimos a json:
    with open(path_fold, 'w', encoding='utf-8') as f:
        json.dump(resultado_json, f, ensure_ascii=False, indent=4)

    print(f"\nArchivo JSON exportado en: {path_fold}")

    if Plotting_Results:
        plt.figure(figsize=(12,6))
        plt.plot(df_resampled['TARIF_PROM'], df_resampled['VENTA'], color='blue', label='VENTA', alpha=0.75)
        plt.plot(df_resampled['TARIF_PROM'], df_resampled['VENTA_Suavizada_Tendencia'], color='red', label='VENTA_Tendencia')
        plt.axvline(x=df_agg[ 'TARIF_PROM' ].mean(), label=f"Costo_Prom = {np.round( df_agg[ 'TARIF_PROM' ].mean(), 0)}", color='orange', linestyle='--', alpha=0.85)
        plt.axvline(x=resultado_opt['P_optimo'], label=f"Tarifa óptima = {np.round( resultado_opt['P_optimo'], 0)}", color='darkgreen', linestyle='--', alpha=0.95)
        plt.axvline(x=df_agg[ 'TARIFA_BASE_TRAMO' ].mean(), label=f"Tarifa Base = {np.round( df_agg[ 'TARIF_PROM' ].mean(), 0)}", color='k', linestyle='--', alpha=0.65)
        plt.legend()
        plt.title('Elasticidad Precio-Demanda Diaria (controlada por hora, anticipación, asiento, mes y día)')
        plt.xlabel('TARIF_PROM')
        plt.ylabel('VENTA')
        plt.grid(True)
        plt.show()

    return [df_agg, df_resampled, resultado_opt]

ruta_principal = os.getcwd()

# Construir la ruta al archivo de configuración "config/config.json".
config_path = os.path.join(ruta_principal, "config", "config.json")
json_Net = os.path.join(ruta_principal, "Files", "ElasAlan.json")

# Llamar a la función externa que carga y realiza preprocesamiento inicial.
Frame = cargar_y_preparar_datos(config_path, ruta_principal)
    
Calculate_Min_Max_Optimal_Tarify( Frame, 365, grados=(1, 2, 3), resampling=True, 
                                 resamp_interval = 10, window=7,
                                        Plotting_Resume=True, Plotting_Resamp=True, 
                                        Plotting_Results=True, Plotting_Elasticity=True,
                                        path_fold=json_Net )