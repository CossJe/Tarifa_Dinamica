import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def year_day_hour_plot( df, colYear, colHour, colDay, colValue, aggfun='sum' ):
    for (anio), data in df.groupby([colYear]):
        pivot = data.pivot_table(
            index=colDay,
            columns=colHour,
            values=colValue,
            aggfunc=aggfun
        )
        
        plt.figure(figsize=(10,4))
        sns.heatmap(pivot, cmap='nipy_spectral', annot=True, fmt='.0f')
        plt.title(f"Ventas por Día y Hora - {anio}")
        plt.xlabel( colHour )
        plt.ylabel( colDay )
        plt.tight_layout()
        plt.show()


def day_hour_plot( df, colHour, colDay, colValue, aggfun='sum' ):
    pivot = df.pivot_table(
        index=colDay,
        columns=colHour,
        values=colValue,
        aggfunc=aggfun
    )
    
    plt.figure(figsize=(10,4))
    sns.heatmap(pivot, cmap='nipy_spectral', annot=True, fmt='.0f')
    plt.title(f"{colValue} por Día y Hora")
    plt.xlabel( colHour )
    plt.ylabel( colDay )
    plt.tight_layout()
    plt.show()


def advance_sales_plot( data, colPrevDays, colVal1, colVal2, xlog=False, ylog=False ):
        # Ordenar por DIAS_ANTICIPACION
        data = data.sort_values( colPrevDays )
        
        #x = np.arange(len(data[ colPrevDays ]))  # Posición de las barras
        width = 0.4  # Ancho de las barras
        
        fig, ax1 = plt.subplots(figsize=(14,6))  # Estira horizontalmente

        x = data[ colPrevDays ]
        y1 = data[ colVal1 ]
        y2 = data[ colVal2 ]

        if xlog == True:
            x = np.log( data[ colPrevDays ] )
        elif ylog == True:
            y1 = np.log( data[ colVal1 ] )
            y2 = np.log( data[ colVal2 ] )

        # Barra de BOLETOS_VEND
        barras1 = ax1.plot( x, y1, color="darkblue", marker="o", linewidth=1.5, label=colVal1 )
        
        # Crear segundo eje Y
        ax2 = ax1.twinx()
        barras2 = ax2.plot( x, y2, color="darkred", marker="o", linewidth=2, label=colVal2 ) 
        
        # Eje X
        ax1.set_xticks(x)
        ax1.set_xticklabels(data[colPrevDays], rotation=90)
        ax1.set_xlabel( colPrevDays )
        
        # Ejes Y
        ax1.set_ylabel( colVal1, color='darkblue')
        ax2.set_ylabel( colVal2, color='darkred')
        
        # Leyenda combinada
        barras = barras1 + barras2
        labels = [b.get_label() for b in barras]
        ax1.legend(barras, labels, loc='upper left')
        
        # Título
        plt.title(f'{colVal1} y {colVal2} por Días de Anticipación')
        plt.grid()
        
        plt.tight_layout()
        plt.show()


def pieplot( df, colClass, colValue ):
    # Gráfico de pastel sobre VENTA
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        df[ colValue ],
        labels=df[ colClass ],
        autopct=lambda p: f"{p:.1f}%\n(${p*df[colValue].sum()/100:,.0f})",  # % + valor absoluto
        startangle=90,
        colors=plt.cm.tab20c.colors,   # paleta de colores
        wedgeprops=dict(width=0.4, edgecolor="w")  # estilo tipo "donut"
    )

    # Ajustes estéticos
    plt.setp(autotexts, size=9, weight="bold", color="black")
    plt.setp(texts, size=10)

    ax.set_title(f"Participación de {colValue} por {colClass}", fontsize=14, weight="bold")
    plt.show()