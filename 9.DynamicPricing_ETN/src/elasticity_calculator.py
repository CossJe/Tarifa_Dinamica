import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
import statsmodels.formula.api as smf


# -------------------------------
# 1. Agrupación y preparación
# -------------------------------
def preparar_dataframe(df):
    df = df.copy()
    df = df[df['VENTA'] > 0]
    df['DIA_SEMANA'] = df['FECHA_OPERACION'].dt.dayofweek
    
    # Agrupar por fecha de corrida (día del viaje)
    df_agrupado = df.groupby(['MES', 'FECHA_CORRIDA', 'DIA_SEMANA', 'HORA_DECIMAL']).agg({
        'TARIFA_BASE_TRAMO': 'mean',
        'PAX_SUBEN': 'sum',
        'VENTA': 'mean',
        'BOLETOS_VEND': 'sum',
        'CAPACIDAD_ASIENTOS_TRAMO': 'mean',
        'DIAS_ANTICIPACION': 'mean',
        'HORAS_ANTICIPACION': 'mean',
        'OCUPACION_TRAMO': 'mean'
    }).reset_index()

    # Renombrar para claridad
    df_agrupado.rename(columns={
        'TARIFA_BASE_TRAMO': 'precio_unitario',
        'PAX_SUBEN': 'demanda_real',
        'VENTA': 'ingreso_real',
        'HORA_DECIMAL' : 'hora',
        'DIA_SEMANA' : 'dia',
        'MES' : 'mes'
    }, inplace=True)

    return df_agrupado

# -------------------------------
# 2. Selección de variables predictoras
# -------------------------------
def seleccionar_predictoras(df):
    correlaciones = df.corr(numeric_only=True)['demanda_real'].sort_values(ascending=False)
    mejores_vars = correlaciones.index[1:4].tolist()  # Excluye 'demanda_real' misma
    return mejores_vars

# -------------------------------
# 3. Cálculo de elasticidad
# -------------------------------
def calcular_elasticidad(df):
    df['log_precio'] = np.log1p(df['precio_unitario'])
    df['log_ingreso'] = np.log1p(df['ingreso_real'])
    df['log_demanda'] = np.log1p(df['demanda_real'])
    # Predicción de demanda
    X = sm.add_constant(df[['mes', 'dia', 'hora', 'log_ingreso']])
    print( X.isnull().sum() )
    y = df['log_demanda']
    modelo = sm.OLS( y, X ).fit()
    elasticidad = np.expm1( modelo.params['log_ingreso'] )
    print( f"Elasticidad = {elasticidad}" )
    return elasticidad, modelo

# -------------------------------
# 4. Gráfico de elasticidad
# -------------------------------
def graficar_elasticidad(df, modelo):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['log_ingreso'], df['demanda_real'], alpha=0.6, label='Datos')
    plt.plot(df['log_ingreso'], modelo.predict(), color='red', label='Regresión')
    plt.xlabel('Log(Precio Real)')
    plt.ylabel('Log(Demanda Real)')
    plt.title('Elasticidad de la Demanda')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 5. Optimización de ingresos
# -------------------------------
def optimizar_precio(df, p0, rango=(100, 2000)):
    def ingresos(p):
        q, modelo = calcular_elasticidad(df)
        return -p * q  # Negativo para maximizar

    resultado = minimize(ingresos, x0=p0, bounds=[rango])
    precio_optimo = resultado.x[0]
    ingreso_maximo = -resultado.fun
    return precio_optimo, ingreso_maximo

# -------------------------------
# 6. Gráfico de ingresos vs precio
# -------------------------------
def graficar_ingresos(Q0, p0, e, rango=(100, 1000)):
    precios = np.linspace(rango[0], rango[1], 100)
    ingresos = [p * Q0 * (p / p0) ** e for p in precios]

    plt.figure(figsize=(8, 5))
    plt.plot(precios, ingresos, label='Ingresos')
    plt.axvline(x=p0, color='gray', linestyle='--', label='Precio Actual')
    plt.axvline(x=precios[np.argmax(ingresos)], color='green', linestyle='--', label='Precio Óptimo')
    plt.xlabel('Precio')
    plt.ylabel('Ingresos Estimados')
    plt.title('Curva de Ingresos vs Precio')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return precios, ingresos

# -------------------------------
# 7. Ejecución completa
# -------------------------------
def ejecutar_modelo(df):
    df_agrupado = preparar_dataframe(df)
    mejores_vars = seleccionar_predictoras(df_agrupado)
    print( df_agrupado.head() )
    elasticidad, modelo = calcular_elasticidad(df_agrupado)
    graficar_elasticidad(df_agrupado, modelo)

    p0 = df_agrupado['precio_unitario'].mean()
    Q0 = df_agrupado['demanda_real'].mean()

    precio_optimo, ingreso_maximo = optimizar_precio(df_agrupado, p0, elasticidad)
    precios, ingresos = graficar_ingresos(Q0, p0, elasticidad)

    resultados = {
        'elasticidad': elasticidad,
        'precio_actual': p0,
        'demanda_promedio': Q0,
        'precio_optimo': precio_optimo,
        'ingreso_maximo': ingreso_maximo,
        'variables_predictoras': mejores_vars
    }

    return resultados