# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:09:18 2025

@author: Jesus Coss
"""
import matplotlib.pyplot as plt
import seaborn as sns

# bibliotecas hechas 
import MainElasticidad as ME
import Tools4PrecioD as TP

TBT, PrecioMaximo, PrecioSugerido, Elas= ME. MainElas()

demanda_predicha_base= 1

resultado_optimizacion = TP. encontrar_precio_optimo(
    demanda_base=demanda_predicha_base,
    tarifa_base=TBT,
    elasticidad=Elas # El valor que obtuvimos en la Fase 2
)

precio_final = resultado_optimizacion['precio_optimo']
ingreso_final = resultado_optimizacion['ingreso_maximo']


# --- PASO E: MOSTRAR RESULTADOS Y GRÁFICO ---
print(f"\nPara un viaje con una demanda base predicha de {demanda_predicha_base:.0f} boletos (a ${TBT:.2f}):")
print("\n--------------------------------------------------")
print(f"  El precio óptimo recomendado es: ${precio_final:.2f}")
print(f"  Con este precio, el ingreso esperado se maximiza a: ${ingreso_final:,.2f}")
print("--------------------------------------------------")

# Graficar
df_simulacion = resultado_optimizacion['datos_simulacion']
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
sns.lineplot(data=df_simulacion, x='precio_probado', y='ingreso_esperado', linewidth=3)
plt.axvline(x=precio_final, color='red', linestyle='--', label=f'Precio Óptimo (${precio_final:.2f})')
plt.axhline(y=ingreso_final, color='red', linestyle='--')
plt.title('Curva de Optimización de Ingresos', fontsize=18)
plt.xlabel('Precio del Boleto ($)', fontsize=12)
plt.ylabel('Ingreso Total Esperado ($)', fontsize=12)
plt.legend()
plt.show()