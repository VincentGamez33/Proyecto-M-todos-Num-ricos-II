import pandas as pd
import matplotlib.pyplot as plt

# --- Cargar datos ---
ruta = '01-Limpieza Datos/resumen_homicidios_dolosos.xlsx'
Nacional_Mensual = pd.read_excel(ruta, sheet_name='Nacional_Mensual')
Estatal_Mensual = pd.read_excel(ruta, sheet_name='Estatal_Mensual')

# --- Diccionario para convertir meses ---
meses = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

# --- Convertir columnas a fechas (con limpieza robusta) ---
# Se aplica el mismo método de limpieza de la vez anterior
Nacional_Mensual['Mes'] = Nacional_Mensual['Mes'].astype(str).str.strip().str.capitalize()
Nacional_Mensual['Mes_Num'] = Nacional_Mensual['Mes'].map(meses)
Nacional_Mensual['Fecha'] = pd.to_datetime(
    Nacional_Mensual['Año'].astype(str) + '-' + Nacional_Mensual['Mes_Num'].astype(str) + '-01',
    errors='coerce'
)
Nacional_Mensual = Nacional_Mensual.dropna(subset=['Fecha'])

# (Procesando también Estatal_Mensual, aunque no se usa en esta gráfica)
Estatal_Mensual['Mes'] = Estatal_Mensual['Mes'].astype(str).str.strip().str.capitalize()
Estatal_Mensual['Mes_Num'] = Estatal_Mensual['Mes'].map(meses)
Estatal_Mensual['Fecha'] = pd.to_datetime(
    Estatal_Mensual['Año'].astype(str) + '-' + Estatal_Mensual['Mes_Num'].astype(str) + '-01',
    errors='coerce'
)
Estatal_Mensual = Estatal_Mensual.dropna(subset=['Fecha'])


# === INICIO DE GRÁFICA Y PERSONALIZACIÓN ===

plt.figure(figsize=(14,14)) # Se mantiene el tamaño (8,8) que pediste
plt.scatter(
    Nacional_Mensual['Fecha'],
    Nacional_Mensual['Total_Victimas'],
    color='purple',
    alpha=0.7,
    label='Total nacional'
)

# === 5. Personalizar la gráfica (FUENTES TAMAÑO 20 Y NEGRITAS) ===
plt.title(
    'Homicidios dolosos - Nacional mensual', 
    fontsize=20, 
    fontweight='bold'  # <-- EN NEGRITA
)
plt.xlabel('Año', fontsize=20, fontweight='bold')      # <-- EN NEGRITA
plt.ylabel('Víctimas', fontsize=20, fontweight='bold') # <-- EN NEGRITA

plt.grid(True, linestyle='--', alpha=0.4)

plt.legend(fontsize=20)  # <-- Tamaño 20

# --- Tamaños de números en los ejes ---
plt.xticks(fontsize=20)  # <-- Tamaño 20
plt.yticks(fontsize=20)  # <-- Tamaño 20
# --------------------------------------

plt.tight_layout()
plt.savefig('01-Limpieza Datos/01-a-Nacional/homicidios_nacional.png')
plt.show()