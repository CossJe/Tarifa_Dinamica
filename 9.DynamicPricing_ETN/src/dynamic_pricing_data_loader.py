import pandas as pd
import numpy as np
import os
import json

from src.data_loader import DIVER_ODBC_Extraction, DB2ConnectorODBC, combine_dfs, optimize_dataframe, preparar_datos
from src.kpis_calculator import ventas_x_dia_hora, calcular_kpis
from src.client_segmentation import smooth_data_dynamic_pricing, decay_line_segment, clasificar_asientos, clasificar_meses, clasificar_dia_hora, merge_with_classification

def cargar_configuracion(config_path):
    """Carga la configuración desde un archivo JSON"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def cargar_y_preparar_datos(config_path, ruta_principal):
    """Función principal para cargar y preparar datos"""
    # Cargar configuración
    config = cargar_configuracion(config_path)
    
    # Construir rutas de archivos SQL
    ruta_queries = os.path.join(ruta_principal, "queries")
    archivos_sql = config["archivos_sql"]
    
    mc_sql_file = os.path.join(ruta_queries, archivos_sql["MC_SQL_File"])
    fik_sql_file = os.path.join(ruta_queries, archivos_sql["FIK_SQL_File"])
    doters_sql_file = os.path.join(ruta_queries, archivos_sql["Doters_SQL_File"])
    db2_sql_file = os.path.join(ruta_queries, archivos_sql["DB2_SQL_File"])
    
    # Convertir listas a formato SQL
    origen_sql = ', '.join(f"'{o}'" for o in config["Ruta"]["ORIG"])
    destino_sql = ', '.join(f"'{d}'" for d in config["Ruta"]["DEST"])
    
    # Leer y procesar consultas SQL
    with open(mc_sql_file, 'r', encoding='utf-8') as file:
        mc_query = file.read().replace("VAR_ORIG_SQL", origen_sql).replace("VAR_DEST_SQL", destino_sql)
    
    with open(fik_sql_file, 'r', encoding='utf-8') as file:
        fik_query = file.read().replace("VAR_ORIG_SQL", origen_sql).replace("VAR_DEST_SQL", destino_sql)
    
    with open(doters_sql_file, 'r', encoding='utf-8') as file:
        dot_query = file.read().replace("VAR_ORIG_SQL", origen_sql).replace("VAR_DEST_SQL", destino_sql)
    
    with open(db2_sql_file, 'r', encoding='utf-8') as file:
        db2_query = file.read().replace("VAR_ORIG_SQL", origen_sql).replace("VAR_DEST_SQL", destino_sql)
    
    # Extraer datos
    mc = DIVER_ODBC_Extraction(config["TLU_MC"], mc_query)
    fik = DIVER_ODBC_Extraction(config["TLU_FIK"], fik_query)
    doters = DIVER_ODBC_Extraction(config["TLU_DOTERS"], dot_query)
    
    # Combinar y optimizar datos
    df_final = combine_dfs(mc, fik, doters)
    df_final = optimize_dataframe(df_final, verbose=True)
    df_final = preparar_datos(df_final)
    
    return df_final

def calcular_kpis_y_clasificar(df_final, año="ALL", año_filtro=None):
    """Función principal para calcular KPIs y clasificar datos"""
    # Calcular KPIs
    kpis = calcular_kpis(df_final, año, año_filtro)
    
    # Segmentación de usuarios por anticipación de compra
    kpis['venta_anticipada'] = smooth_data_dynamic_pricing(kpis['venta_anticipada'], 'BOLETOS_VEND')
    kpis['venta_anticipada'], segmentos = decay_line_segment(
        kpis['venta_anticipada'], 'DIAS_ANTICIPACION', 'BOLETOS_VEND_SMOOTH', max_segmentos=4, verbose=False
    )
    
    # Clasificar asientos, meses y día/hora
    kpis['venta_x_asientos'] = clasificar_asientos(kpis['venta_x_asientos'])
    kpis['ventas_por_mes'] = clasificar_meses(kpis['ventas_por_mes'], col_pax="PAX_SUBEN", anio_col="AÑO", mes_col="MES")
    kpis['ocupacion_promedio'] = clasificar_dia_hora(
        kpis['ocupacion_promedio'], dia_col="NOMBRE_DIA_CORRIDA", hora_col="HORA_DECIMAL", col2var="FOP_PROM"
    )
    
    # Unir clasificaciones al dataframe principal
    df_final = merge_with_classification(df_final, kpis['venta_anticipada'], merge_on=['DIAS_ANTICIPACION'], cols_merge=['Clasif_venta_anticip'])
    df_final = merge_with_classification(df_final, kpis['venta_x_asientos'], merge_on=['CAPACIDAD_ASIENTOS_TRAMO', 'NUM_ASIENTO'], cols_merge=['Clasif_asiento'])
    df_final = merge_with_classification(df_final, kpis['ventas_por_mes'], merge_on=['AÑO', 'MES'], cols_merge=['Clasif_mes'])
    df_final = merge_with_classification(df_final, kpis['ocupacion_promedio'], merge_on=['NOMBRE_DIA_CORRIDA', 'HORA_DECIMAL'], cols_merge=["Clasif_dia_venta", "Clasif_hora_venta"])
    
    return df_final, kpis