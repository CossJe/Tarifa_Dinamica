# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 12:32:41 2025

@author: Jesus Coss
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier # Importamos el clasificador de XGBoost
from sklearn.preprocessing import StandardScaler # Para escalar, si es necesario

def GetClusters():
    try:
        df = pd.read_excel("ClusteringClientes.xlsx")
    except FileNotFoundError:
        print("Error: Asegúrate de que 'ClusteringClientes.xlsx' esté en el directorio correcto.")
        exit()
    return df
    
def ClusteringSupervisado():
    df= GetClusters()
    # 1. Limpieza y Definición de X y y
    X = df.drop(['EMAIL', 'Cluster'], axis=1) # Excluir EMAIL y ambas columnas de cluster
    y = df['Cluster']
    num_clases_final = y.nunique()
    # 2. Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )
    print(f"Datos listos. Clases finales: {num_clases_final}. Filas de entrenamiento: {X_train.shape[0]}")
    
    
    # --- 2. Entrenamiento del Modelo XGBoost Básico ---
    
    xgb_model_baseline = XGBClassifier(
        objective='multi:softmax',
        num_class=num_clases_final,
        eval_metric='mlogloss',
        n_estimators=100,
        random_state=42
    )
    
    print("\nIniciando entrenamiento básico de XGBoost...")
    xgb_model_baseline.fit(X_train, y_train)
    y_pred_baseline = xgb_model_baseline.predict(X_test)
    
    
    # --- 3. Evaluación del Modelo Básico ---
    
    print("\n" + "="*50)
    print("--- Evaluación del Modelo Básico ---")
    
    accuracy = accuracy_score(y_test, y_pred_baseline)
    print(f"Precisión (Accuracy) General: {accuracy:.4f}")
    # guardado del modelo de prediccion
    xgb_model_baseline.save_model("modelo_xgboost_clientes.json")
