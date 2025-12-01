import pandas as pd
import matplotlib.pyplot as plt
import os

# === 1. Definir rutas y meses ===
# IMPORTANTE: Esta es la ruta del archivo que generó tu script de limpieza
input_file = '01-Limpieza Datos/01-b-Estatal/estatal_mensual_seleccion.xlsx'

# Diccionario para convertir meses
meses = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

# === 2. Función reutilizable para procesar fechas ===
# La usaremos para cada hoja que leamos
def procesar_fechas(df):
    """Toma un DataFrame y le añade la columna 'Fecha'."""
    df_proc = df.copy() 
    # Asegura que la columna Mes se maneje como string
    df_proc['Mes'] = df_proc['Mes'].astype(str).str.strip().str.capitalize()
    df_proc['Mes_Num'] = df_proc['Mes'].map(meses)
    df_proc['Fecha'] = pd.to_datetime(
        df_proc['Año'].astype(str) + '-' + df_proc['Mes_Num'].astype(str) + '-01',
        errors='coerce' # Maneja errores si el mapeo falla
    )
    df_proc = df_proc.dropna(subset=['Fecha']) # Elimina filas con fechas nulas
    return df_proc

# === 3. Cargar TODAS las hojas del Excel ===
try:
    # sheet_name=None carga todas las hojas en un diccionario.
    dict_dfs = pd.read_excel(input_file, sheet_name=None)
    print(f"Archivo '{input_file}' cargado. Hojas encontradas: {list(dict_dfs.keys())}")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{input_file}'.")
    print("Asegúrate de haber ejecutado primero el script de limpieza de datos.")
    exit()

except Exception as e:
    print(f"Error al leer el archivo Excel: {e}")
    exit()

# === 4. Bucle para procesar y graficar cada hoja ===
# Itera sobre cada par (nombre_hoja, dataframe) en el diccionario
for nombre_hoja, df in dict_dfs.items():
    print(f"Procesando hoja: '{nombre_hoja}'...")
    
    # 1. Procesar las fechas
    df_procesado = procesar_fechas(df)
    
    if df_procesado.empty:
        print(f"  [ADVERTENCIA] No hay datos válidos para graficar en '{nombre_hoja}'. Saltando...")
        continue

    # 2. Extraer información del nombre de la hoja
    partes = nombre_hoja.split('_', 1)
    tipo_indice = partes[0] # "Mayor", "Menor" o "Medio"
    nombre_estado = partes[1] # "Guanajuato", "Yucatan", etc.

    # 3. Asignar un color según el tipo
    colores = {'Mayor': 'darkred', 'Menor': 'green', 'Medio': 'royalblue'}
    color_grafica = colores.get(tipo_indice, 'black') # 'black' por si acaso

    # 4. Crear la gráfica
    plt.figure(figsize=(14, 14))
    plt.scatter(
        df_procesado['Fecha'],
        df_procesado['Total_Victimas'],
        color=color_grafica,
        alpha=0.7,
        label=f'Homicidios en {nombre_estado}'
    )

    # === 5. Personalizar la gráfica (FUENTES TAMAÑO 20 Y NEGRITAS) ===
    plt.title(
        f'Homicidios dolosos - {nombre_estado} (Índice {tipo_indice})', 
        fontsize=20,
        fontweight='bold'  # <-- EN NEGRITA
    )
    plt.xlabel('Año', fontsize=20, fontweight='bold')      # <-- EN NEGRITA
    plt.ylabel('Víctimas', fontsize=20, fontweight='bold') # <-- EN NEGRITA
    
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.legend(fontsize=20)  # <-- Tamaño 20
    
    # --- Añade estas líneas para los números de los ejes ---
    plt.xticks(fontsize=20)  # <-- Tamaño 20
    plt.yticks(fontsize=20)  # <-- Tamaño 20
    # -----------------------------------------------------
    
    plt.tight_layout()
    # Fijar el mínimo del eje Y en 0
    plt.ylim(bottom=0)

    # 6. Guardar la gráfica
    # Crea un nombre de archivo único, ej: "grafica_Mayor_Guanajuato.png"
    nombre_archivo = f'01-Limpieza Datos/01-b-Estatal/grafica_{nombre_hoja}.png'
    plt.savefig(nombre_archivo)
    print(f"--- Gráfica guardada como: '{nombre_archivo}'")

    # 7. Cerrar la figura actual
    # Importante para que la siguiente gráfica del bucle se dibuje en un lienzo nuevo.
    plt.close()

print("\nProceso completado. Se generaron 3 gráficas (una para cada hoja del Excel).")