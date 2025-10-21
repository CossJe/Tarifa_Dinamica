# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:00:06 2025

@author: Jesus Coss
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import model_from_json


def Prepare_Data(df):
    #df=Get_Data()
    # Se filtra el DataFrame para incluir solo ventas mayores que cero.
    df = df[df['VENTA'] > 0]
    df=df[df['BOLETOS_VEND']>0]
    df=df.drop('BOLETOS_VEND',axis=1)
    df['FECHA_OPERACION'] = pd.to_datetime(df['FECHA_OPERACION'])
    fecha_maxima = df['FECHA_OPERACION'].max()
    df = df[df['FECHA_OPERACION'] < fecha_maxima].copy()
    df['FECHA_CORRIDA'] = pd.to_datetime(df['FECHA_CORRIDA'])
    
    df["HORA_SALIDA_CORRIDA"] = pd.to_datetime(df["HORA_SALIDA_CORRIDA"])
    
    df['TBT']= df['TARIFA_BASE_TRAMO']-df['IVA_TARIFA_BASE_TRAMO']
    df['%_dif_TBT_Venta']= (df['TBT']-df['VENTA'])/df['TBT']
    df['TIPO_CLASE'] = np.where(
        df['CLASE_SERVICIO'].astype(str).str.contains('DOS PISOS', case=False, na=False),
        'DOS',
        'UNO'
    )
    return df 

def Data4RedNeuronal(df_1):
    df=Prepare_Data(df_1)
    df_total= pd.DataFrame()
    df_total['Origen-Destino'] = df['ORIGEN'].astype(str) + '-' + df['DESTINO'].astype(str)
    df_total['DiaSemana_Corrida']=df['FECHA_CORRIDA'].dt.dayofweek
    df_total['Hora_Corrida']=df['HORA_SALIDA_CORRIDA'].dt.hour
    df_total[['NUM_ASIENTO','HORAS_ANTICIPACION','%_dif_TBT_Venta']]=df[['NUM_ASIENTO','HORAS_ANTICIPACION','%_dif_TBT_Venta']].copy()
    df_total['Mes_Corrida']=df['FECHA_CORRIDA'].dt.month
    df_total['Anio_Corrida']=df['FECHA_CORRIDA'].dt.year
    df_total['Buen_Dia'] = df['FECHA_CORRIDA'].dt.dayofweek.isin([4,5,6,0]).astype(int)
    df_total['Buena_Hora'] = df['HORA_SALIDA_CORRIDA'].dt.hour.isin([23,17,18,19,20]).astype(int)
    df_total['Buen_Mes'] = df['FECHA_CORRIDA'].dt.month.isin([3,4,5,6]).astype(int)
    df_total['Buen_Asiento'] = df['NUM_ASIENTO'].isin([1,2,3,4,5,6,7,8,9,10]).astype(int)
    # Crea un nuevo DataFrame con las variables dummy (codificación one-hot)
    df_dummies = pd.get_dummies(
        df['TIPO_CLIENTE'],
        prefix='TIPO_CLIENTE', # Prefijo para las nuevas columnas (ej: TIPO_CLIENTE_A)
        drop_first=False        # Elimina la primera categoría para evitar multicolinealidad
    ).astype(int)
    
    # Crea un nuevo DataFrame con las variables dummy (codificación one-hot)
    df_dummies1 = pd.get_dummies(
        df['TIPO_CLASE'],
        prefix='PISO', 
        drop_first=False
    ).astype(int)
    
    # Une las nuevas columnas dummy al DataFrame original
    df_total = pd.concat([df_total, df_dummies,df_dummies1], axis=1)
    df_total['VENTA']=df['VENTA'].copy()

    return df_total


def GetFlag(datos):

    asimetria_pandas = datos.skew()
    #print(f"Coeficiente de Asimetría: {asimetria_pandas:.4f}")
    
    if asimetria_pandas > 1.0:
        #print("La asimetría es alta (> 1.0). La transformación logarítmica es altamente recomendable.")
        Bandera=True
    elif asimetria_pandas > 0.5:
        #print("La asimetría es moderada (> 0.5). La transformación logarítmica podría ser beneficiosa.")
        Bandera=True
    else:
        #print("La asimetría es baja. Una transformación no es necesaria.")
        Bandera=False
        
    return Bandera

def GetTrainingForm(df,Bandera):
    # Definir la variable objetivo (Y)
    Y = df['VENTA']
    
    if Bandera:
        # Aplicar la transformación logarítmica a Y
        Y_log = np.log(Y)
    else:
        Y_log = Y.copy()
    
    # Eliminar la variable VENTA del dataframe de features (X)
    X = df.drop('VENTA', axis=1) 
    
    categorical_features= 'Origen-Destino'
    df_ohe = pd.get_dummies(X[categorical_features]).astype(int)
    
    # Columnas numéricas que necesitan Estandarización
    # Excluimos las binarias/dummies que ya están bien escaladas (0 o 1)
    numeric_features = [
        'DiaSemana_Corrida', 'Hora_Corrida', 'NUM_ASIENTO', 
        'HORAS_ANTICIPACION', '%_dif_TBT_Venta', 'Mes_Corrida','Anio_Corrida'
    ]
    
    # Columnas binarias (se dejan pasar sin transformación)
    binary_features = [col for col in X.columns if col not in [categorical_features] + numeric_features]
    
    indice_correcto = X[numeric_features].index # o df_ohe.index
    
    scaler = StandardScaler()
    # 2. Convierte el array escalado (NumPy) a DataFrame, ASIGNANDO el índice correcto
    X_escalado_array = scaler.fit_transform(X[numeric_features])
    X_escalado = pd.DataFrame(X_escalado_array, 
                              index=indice_correcto, # <-- ¡CLAVE!
                              columns=numeric_features)
    
    X_processed= pd.concat([df_ohe, X_escalado,X[binary_features]], axis=1)

    return X_processed, Y_log

def TrainingNet(X_processed,Y_log,Bandera):
    # División del 80% para entrenamiento y 20% para prueba
    X_train, X_test, Y_train_log, Y_test_log = train_test_split(
        X_processed, 
        Y_log, 
        test_size=0.2, 
        random_state=42 # Para asegurar resultados reproducibles
    )
    
    input_feature_count = X_train.shape[1] 
    
    
    # --- 1. Definir el número de features de entrada ---
    # (Esto debe ser el número de columnas de tu X_train después del OHE y estandarización)
    input_shape = X_train.shape[1] 
    
    # --- 2. CONSTRUCCIÓN DEL MODELO ---
    model = Sequential([
        # Capa Oculta 1
        Dense(128, activation='relu', input_shape=(input_shape,)),
        
        # Capa Oculta 2 (Regularización para evitar overfitting)
        # Aquí puedes añadir 'Dropout' si notas que el modelo se sobreajusta
        Dense(64, activation='relu'), 
        
        # Capa de Salida: 1 neurona y activación lineal para regresión
        Dense(1, activation='linear') 
    ])
    
    # --- 3. COMPILACIÓN DEL MODELO ---
    model.compile(
        optimizer='adam',
        loss='mse',           # Función de pérdida: Error Cuadrático Medio
        metrics=['mae', 'mse']  # Métricas a monitorear: MAE y MSE
    )
    
    # --- 4. ENTRENAMIENTO (Ejemplo) ---
    history = model.fit(
        X_train, 
        Y_train_log,  # ¡Usamos la variable VENTA transformada con logaritmo!
        epochs=2, 
        batch_size=32, 
        validation_split=0.2, # Usamos el 20% para validación interna
        verbose=1
    )
    
    PredictingNet(model,X_test, Y_test_log,Bandera)
    return model

def PredictingNet(model,X_test, Y_test_log,Bandera):
    # 'model' es tu red neuronal entrenada
    # 'X_test' son tus features de prueba (escalados y codificados)
    Y_pred_log = model.predict(X_test)
    
    if Bandera:
        # Revertir la predicción logarítmica a la escala de precio real
        Y_pred_real = np.exp(Y_pred_log)
        # Revertir los valores reales de prueba (Y_test_log) a la escala de precio real
        # Esto es para compararlos directamente
        Y_test_real = np.exp(Y_test_log) 
    else:
        Y_pred_real= Y_pred_log.copy()
        Y_test_real = Y_test_log.copy()
    
    
    # Calcular el MAE real
    mae_real = mean_absolute_error(Y_test_real, Y_pred_real)
    
    print(f"\nEl Error Absoluto Medio (MAE) final es de: {mae_real:,.2f} [Moneda]")
    
    #  Calcular el Error Cuadrático Medio (MSE)
    mse_real = mean_squared_error(Y_test_real, Y_pred_real)
    
    # Calcular la Raíz del Error Cuadrático Medio (RMSE)
    #    La RMSE es simplemente la raíz cuadrada del MSE
    rmse_real = np.sqrt(mse_real)
    
    print(f"\nLa Raíz del Error Cuadrático Medio (RMSE) final es de: {rmse_real:,.2f} [Moneda]")
    
# ------------------------------------------------------------------------

#                       Fase para las predicciones 

# -------------------------------------------------------------------------

def DataForecasting(df,datos_carac):

    df['TIPO_CLASE'] = np.where(
        df['CLASE_SERVICIO'].astype(str).str.contains('DOS PISOS', case=False, na=False),
        'DOS',
        'UNO'
    )
    df["HORA_SALIDA_CORRIDA"] = pd.to_datetime(df["HORA_SALIDA_CORRIDA"])
    df['FECHA_CORRIDA'] = pd.to_datetime(df['FECHA_CORRIDA'])
    df_total = pd.DataFrame(columns=datos_carac["FrameN.columns"])
    
    df_total['Origen-Destino'] = df['ORIGEN'].astype(str) + '-' + df['DESTINO'].astype(str)
    df_total['DiaSemana_Corrida']=df['FECHA_CORRIDA'].dt.dayofweek
    df_total['Hora_Corrida']=df['HORA_SALIDA_CORRIDA'].dt.hour
    df_total[['NUM_ASIENTO','HORAS_ANTICIPACION']]=df[['NUM_ASIENTO','HORAS_ANTICIPACION']].copy()
    df_total['%_dif_TBT_Venta']=datos_carac['%_dif_TBT_Venta']
    df_total['Mes_Corrida']=df['FECHA_CORRIDA'].dt.month
    df_total['Anio_Corrida']=df['FECHA_CORRIDA'].dt.year
    df_total['Buen_Dia'] = df['FECHA_CORRIDA'].dt.dayofweek.isin([4,5,6,0]).astype(int)
    df_total['Buena_Hora'] = df['HORA_SALIDA_CORRIDA'].dt.hour.isin([23,17,18,19,20]).astype(int)
    df_total['Buen_Mes'] = df['FECHA_CORRIDA'].dt.month.isin([3,4,5,6]).astype(int)
    df_total['Buen_Asiento'] = df['NUM_ASIENTO'].isin([1,2,3,4,5,6,7,8,9,10]).astype(int)
    # Crea un nuevo DataFrame con las variables dummy (codificación one-hot)
    df_dummies = pd.get_dummies(
        df['TIPO_CLIENTE'],
        prefix='TIPO_CLIENTE', # Prefijo para las nuevas columnas (ej: TIPO_CLIENTE_A)
        drop_first=False        # Elimina la primera categoría para evitar multicolinealidad
    ).astype(int)

    df_total[df_dummies.columns]= df_dummies[df_dummies.columns].copy()
    # Crea un nuevo DataFrame con las variables dummy (codificación one-hot)
    df_dummies1 = pd.get_dummies(
        df['TIPO_CLASE'],
        prefix='PISO', 
        drop_first=False
    ).astype(int)

    df_total[df_dummies1.columns]= df_dummies1[df_dummies1.columns].copy()
    # Une las nuevas columnas dummy al DataFrame original

    df_total['VENTA']=df['VENTA'].copy()

    #df_total=df_total.fillna(0)
    
    return df_total

def PrepareData4Fore(df):
    #df=Get_Data()
    # Se filtra el DataFrame para incluir solo ventas mayores que cero.
    df = df[df['VENTA'] > 0]
    df['FECHA_OPERACION'] = pd.to_datetime(df['FECHA_OPERACION'])
    fecha_maxima = df['FECHA_OPERACION'].max()
    df = df[df['FECHA_OPERACION'] == fecha_maxima].copy()
    return df 

def GetPredictingForm(Fore,cols):
    X1 = Fore.drop('VENTA', axis=1)
    
    X_final=pd.DataFrame(columns=cols)
    
    categorical_features= 'Origen-Destino'
    df_ohe = pd.get_dummies(X1[categorical_features]).astype(int)
    
    # Columnas numéricas que necesitan Estandarización
    # Excluimos las binarias/dummies que ya están bien escaladas (0 o 1)
    numeric_features = [
        'DiaSemana_Corrida', 'Hora_Corrida', 'NUM_ASIENTO', 
        'HORAS_ANTICIPACION', '%_dif_TBT_Venta', 'Mes_Corrida','Anio_Corrida'
    ]
    
    # Columnas binarias (se dejan pasar sin transformación)
    binary_features = [col for col in X1.columns if col not in [categorical_features] + numeric_features]
    
    indice_correcto = X1[numeric_features].index # o df_ohe.index
    
    scaler = StandardScaler()
    # 2. Convierte el array escalado (NumPy) a DataFrame, ASIGNANDO el índice correcto
    X_escalado_array = scaler.fit_transform(X1[numeric_features])
    X_escalado = pd.DataFrame(X_escalado_array, 
                              index=indice_correcto, # <-- ¡CLAVE!
                              columns=numeric_features)
    
    X_processed1= pd.concat([df_ohe, X_escalado,X1[binary_features]], axis=1)
    
    X_final[X_processed1.columns]= X_processed1[X_processed1.columns].copy()
    X_final=X_final.fillna(0)

    return X_final

def GetValues(model,Fore,X_final,Bandera):
    # 'model' es tu red neuronal entrenada
    # 'X_test' son tus features de prueba (escalados y codificados)
    Y_pred_log = model.predict(X_final)
    
    Y_R_real = Fore['VENTA']
                
    if Bandera:
        # Revertir la predicción logarítmica a la escala de precio real
        Y_pred_real = np.exp(Y_pred_log)
    else:
        Y_pred_real= Y_pred_log.copy()
    
    # Calcular el MAE real
    mae_real = mean_absolute_error(Y_R_real, Y_pred_real)
    
    #print(f"\nEl Error Absoluto Medio (MAE) final es de: {mae_real:,.2f} [Moneda]")
    return Y_pred_real
    
def ProcessingNet(data):
    # Obtener el directorio de trabajo actual (ruta principal del proyecto).
    ruta_principal = os.getcwd()

    # Construir la ruta al archivo 
    json_path = os.path.join(ruta_principal, "Models", "modelo_arquitectura.json")
    json_Net = os.path.join(ruta_principal, "Files", "caracNet.json")
    weights_path = os.path.join(ruta_principal, "Models", "modelo_pesos.weights.h5")
    
    df=data[data.columns[1:]]
    Frame=df.copy()
    FrameN=Data4RedNeuronal(Frame.copy())
    Bandera = GetFlag(FrameN['VENTA'])
    X_processed, Y_log = GetTrainingForm(FrameN.copy(),Bandera)
    model= TrainingNet(X_processed,Y_log,Bandera)
    
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(weights_path)
    
    carac = {
        "X_processed.columns": list(X_processed.columns),
        "Bandera": Bandera,
        "%_dif_TBT_Venta": float(FrameN['%_dif_TBT_Venta'].mean()),
        "FrameN.columns": list(FrameN.columns)
    }

    
    with open(json_Net, "w", encoding="utf-8") as f:
        json.dump(carac, f, ensure_ascii=False, indent=4)
    
    return

def NewClientsPredNet(data):
    # Obtener el directorio de trabajo actual (ruta principal del proyecto).
    ruta_principal = os.getcwd()

    # Construir la ruta al archivo 
    json_path = os.path.join(ruta_principal, "Models", "modelo_arquitectura.json")
    json_Net = os.path.join(ruta_principal, "Files", "caracNet.json")
    weights_path = os.path.join(ruta_principal, "Models", "modelo_pesos.weights.h5")
    
    with open(json_Net, 'r') as f:
        # 2. Cargar el contenido del archivo JSON
        datos_carac = json.load(f)
            
    # 1. Cargar la arquitectura desde JSON
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    
    loaded_model = model_from_json(loaded_model_json)
    
    # 2. Cargar los pesos entrenados desde HDF5
    loaded_model.load_weights(weights_path)
    
    # 3. Compilar el modelo cargado (necesario antes de hacer predicciones)
    loaded_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    df_= PrepareData4Fore(data.copy())
    df_today=df_[df_.columns[1:]]
    Fore=DataForecasting(df_today,datos_carac)
    X_final= GetPredictingForm(Fore, datos_carac["X_processed.columns"])
    PrecioDin=GetValues(loaded_model,Fore,X_final,datos_carac["Bandera"])
    df_["PRECIO DINAMICO"]= PrecioDin
    
    return df_