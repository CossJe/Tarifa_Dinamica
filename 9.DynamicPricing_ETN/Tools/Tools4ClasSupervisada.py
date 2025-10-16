# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:34:32 2025

@author: Jesus Coss
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier # Importamos el clasificador de XGBoost
from sklearn.preprocessing import StandardScaler # Para escalar, si es necesario
import os
import json
 
def ClusteringSupervisado(df):
    # 1. Limpieza y Definición de X y y
    X = df.drop(['EMAIL', 'Cluster'], axis=1) # Excluir EMAIL y ambas columnas de cluster
    y = df['Cluster']
    num_clases_final = y.nunique()
    # 2. Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )
 
    # --- 3. Entrenamiento del Modelo XGBoost Básico ---
    
    xgb_model_baseline = XGBClassifier(
        objective='multi:softmax',
        num_class=num_clases_final,
        eval_metric='mlogloss',
        n_estimators=100,
        random_state=42
    )
    
    xgb_model_baseline.fit(X_train, y_train)
    y_pred_baseline = xgb_model_baseline.predict(X_test)

    # guardado del modelo de prediccion
    
    ruta_principal = os.getcwd()
    nombre_archivo= "modelo_xgboost_clientes.json"
    config_path = os.path.join(ruta_principal, "Models", nombre_archivo)
    xgb_model_baseline.save_model(config_path)
    1
    print(f"Modelo XGBoost guardado exitosamente como: {nombre_archivo}")