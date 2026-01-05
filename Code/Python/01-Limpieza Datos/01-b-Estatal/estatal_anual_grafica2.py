import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob

CARPETA_DATOS = '01-Limpieza Datos/Archivos Limpios Excel'

def obtener_datos(patron):
    busqueda = os.path.join(CARPETA_DATOS, f"{patron}*.xlsx")
    archivos = glob.glob(busqueda)
    if archivos:
        df = pd.read_excel(archivos[0])
        nombre = os.path.basename(archivos[0]).replace('.xlsx', '').split('_', 1)[1]
        meses_num = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
        df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].map(meses_num).astype(str) + '-01')
        return df.sort_values('Fecha'), nombre
    return None, None

# Lista de configuraciones para iterar
configuraciones = [
    ("Mayor_", "#c62828", "Mayor Índice"),
    ("Medio_", "#f9a825", "Índice Medio"),
    ("Menor_", "#2e7d32", "Menor Índice")
]

# ========================================================
# BUCLE DE GENERACIÓN DE GRÁFICAS INDIVIDUALES
# ========================================================
for patron, color, titulo_tipo in configuraciones:
    df, nombre_estado = obtener_datos(patron)
    
    if df is not None:
        print(f"Generando gráfica para: {nombre_estado} ({titulo_tipo})...")
        
        plt.figure(figsize=(14, 7))
        
        # Estilo Puntos + Línea
        plt.plot(df['Fecha'], df['Victimas'], color=color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=1)
        plt.scatter(df['Fecha'], df['Victimas'], color=color, s=35, alpha=0.8, zorder=2)
        
        # Ejes
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Títulos Dinámicos
        plt.title(f'Evolución Mensual: {nombre_estado} ({titulo_tipo})', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Año', fontsize=12)
        plt.ylabel('Víctimas', fontsize=12)
        
        plt.grid(axis='y', linestyle='-', alpha=0.2, color='gray')
        plt.grid(axis='x', linestyle=':', alpha=0.1)
        
        plt.tight_layout()
        
        # Guardar con nombre específico
        nombre_archivo = f"grafica_{nombre_estado}.pdf"
        plt.savefig(nombre_archivo)
        print(f" -> Guardada: {nombre_archivo}")
        plt.close() # Cerrar para liberar memoria

print("Proceso terminado.")
