import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ========================================================
# CONFIGURACIÓN
# ========================================================
# Asegúrate de que esta ruta sea correcta en tu PC
CARPETA_DATOS = '01-Limpieza Datos/Archivos Limpios Excel'
ARCHIVO_NACIONAL = os.path.join(CARPETA_DATOS, 'Nacional.xlsx')

# ========================================================
# 1. CARGA Y CREACIÓN DE FECHA
# ========================================================
print(f"--- Generando gráfica Nacional (Puntos y Líneas) ---")
try:
    df = pd.read_excel(ARCHIVO_NACIONAL)
    
    # Mapa de meses
    meses_num = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
        'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    
    # Crear columna de fecha
    df['Mes_Num'] = df['Mes'].map(meses_num)
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01')
    
    # Ordenar cronológicamente (CRUCIAL para que la línea no haga garabatos)
    df = df.sort_values('Fecha')

except FileNotFoundError:
    print(f"Error: No se encontró {ARCHIVO_NACIONAL}")
    exit()

# ========================================================
# 2. GRAFICACIÓN (ESTILO ELEGANTE CONECTADO)
# ========================================================
plt.figure(figsize=(14, 8))

# Color principal (Morado oscuro elegante)
MAIN_COLOR = '#6a1b9a'

# --- A) La Línea Conectora ---
# La dibujamos primero (zorder=1) y más transparente (alpha=0.4)
# para que quede sutilmente en el fondo.
plt.plot(df['Fecha'], df['Victimas'], 
         color=MAIN_COLOR,
         linestyle='-',       # Línea sólida
         linewidth=1.5,       # Grosor moderado
         alpha=0.4,           # Semitransparente para no saturar
         zorder=1)            # Capa inferior

# --- B) Los Puntos (Scatter) ---
# Los dibujamos encima (zorder=2) y más sólidos (alpha=0.8)
# para que sean los protagonistas.
plt.scatter(df['Fecha'], df['Victimas'], 
            color=MAIN_COLOR,
            s=35,             # Tamaño del punto ligeramente aumentado
            alpha=0.8,        # Más opaco que la línea
            zorder=2,         # Capa superior
            label='Nacional Mensual')

# Definir AX
ax = plt.gca()

# Configuración del Eje X (Fechas)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Títulos y Grid
plt.title('Evolución Mensual de Homicidios Nacionales (2015-2025)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Víctimas', fontsize=12)

# Grid minimalista
plt.grid(axis='y', linestyle='-', alpha=0.2, color='gray') 
plt.grid(axis='x', linestyle=':', alpha=0.1)

plt.legend(loc='upper left')
plt.tight_layout()

print("Guardando imagen...")
plt.savefig('grafica_Nacional.pdf')
print("Gráfica generada: grafica_Nacional.pdf")
plt.show()