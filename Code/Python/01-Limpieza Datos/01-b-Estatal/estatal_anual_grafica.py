import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob

# ========================================================
# CONFIGURACIÓN
# ========================================================
CARPETA_DATOS = '01-Limpieza Datos/Archivos Limpios Excel'

def obtener_datos(patron):
    busqueda = os.path.join(CARPETA_DATOS, f"{patron}*.xlsx")
    archivos = glob.glob(busqueda)
    if archivos:
        df = pd.read_excel(archivos[0])
        nombre = os.path.basename(archivos[0]).replace('.xlsx', '').split('_', 1)[1]
        
        meses_num = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
        df['Mes_Num'] = df['Mes'].map(meses_num)
        df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01')
        return df.sort_values('Fecha'), nombre
    return None, None

# Cargar
df1, n1 = obtener_datos("Mayor_")
df2, n2 = obtener_datos("Medio_")
df3, n3 = obtener_datos("Menor_")

# ========================================================
# GRAFICACIÓN
# ========================================================
plt.figure(figsize=(15, 8))

# Función auxiliar para graficar estilo "Puntos + Línea"
def plot_estilo(df, nombre, color, etiqueta_prefijo):
    if df is not None:
        # Línea de fondo
        plt.plot(df['Fecha'], df['Victimas'], color=color, linestyle='-', linewidth=1.5, alpha=0.4, zorder=1)
        # Puntos encima
        plt.scatter(df['Fecha'], df['Victimas'], color=color, s=30, alpha=0.8, zorder=2, label=f'{etiqueta_prefijo}: {nombre}')

# 1. Mayor (Rojo oscuro)
plot_estilo(df1, n1, '#c62828', 'Mayor')

# 2. Medio (Ámbar/Naranja)
plot_estilo(df2, n2, '#f9a825', 'Medio')

# 3. Menor (Verde bosque)
plot_estilo(df3, n3, '#2e7d32', 'Menor')

# Configuración Ejes
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Decoración
plt.title('Comparativa Estatal Mensual: Trayectorias Completas (2015-2025)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Víctimas Mensuales', fontsize=12)
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Grid
plt.grid(axis='y', linestyle='-', alpha=0.2, color='gray')
plt.grid(axis='x', linestyle=':', alpha=0.1)

plt.tight_layout()
plt.savefig('grafica_estatal_comparativa.pdf')
print("Gráfica generada: grafica_estatal_comparativa.pdf")
plt.show()
