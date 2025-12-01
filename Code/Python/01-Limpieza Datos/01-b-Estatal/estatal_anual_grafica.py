import pandas as pd
import plotly.express as px
import os

# === 1. Definir rutas y meses ===
input_file = '01-Limpieza Datos/01-b-Estatal/estatal_mensual_seleccion.xlsx'

# Diccionario para convertir meses
meses = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

# === 2. Función reutilizable para procesar fechas (VERSIÓN CORREGIDA) ===
def procesar_fechas(df):
    """Toma un DataFrame y le añade la columna 'Fecha'."""
    df_proc = df.copy()
    
    try:
        # --- INICIO DE LA CORRECCIÓN ---
        # 1. Limpia la columna 'Mes' de espacios y la pone en formato 'Capitalizado'
        df_proc['Mes_Limpio'] = df_proc['Mes'].str.strip().str.capitalize()
        
        # 2. Mapea usando la columna limpia
        df_proc['Mes_Num'] = df_proc['Mes_Limpio'].map(meses)
        
        # 3. Comprueba si hubo errores en el mapeo (mostrará si hay valores malos)
        if df_proc['Mes_Num'].isnull().any():
            print(f"  [ADVERTENCIA] Se encontraron valores de 'Mes' no válidos. Filas con error:")
            # Muestra las filas donde 'Mes' no se pudo mapear
            print(df_proc[df_proc['Mes_Num'].isnull()][['Año', 'Mes']])

        # 4. Crea la columna de fecha
        df_proc['Fecha'] = pd.to_datetime(
            df_proc['Año'].astype(str) + '-' + df_proc['Mes_Num'].astype(str) + '-01',
            errors='coerce' # 'coerce' convierte errores en NaT (Not a Time)
        )
        
        # 5. Comprueba si hay fechas fallidas y elimina esas filas
        if df_proc['Fecha'].isnull().any():
            print("  [ERROR] Falló la conversión a pd.to_datetime. Se eliminarán estas filas de la gráfica.")
            df_proc = df_proc.dropna(subset=['Fecha']) # Elimina filas con NaT
            
        # --- FIN DE LA CORRECCIÓN ---
            
    except Exception as e:
        print(f"  [ERROR] Fallo crítico en procesar_fechas: {e}")
        print("  Asegúrate que las columnas 'Año' y 'Mes' existen y son correctas.")
        return None # Devuelve None para que el bucle no intente graficar

    return df_proc

# === 3. Cargar TODAS las hojas del Excel ===
try:
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
for nombre_hoja, df in dict_dfs.items():
    
    print(f"Procesando hoja: '{nombre_hoja}'...")
    
    # 1. Procesar las fechas (usando la nueva función robusta)
    df_procesado = procesar_fechas(df)
    
    # Si la función falló, salta a la siguiente hoja
    if df_procesado is None or df_procesado.empty:
        print(f"  [ERROR] No se pudo procesar la hoja '{nombre_hoja}'. Saltando...")
        continue # <-- IMPORTANTE

    # 2. Extraer información del nombre de la hoja
    partes = nombre_hoja.split('_', 1)
    tipo_indice = partes[0]
    nombre_estado = partes[1]

    # 3. Asignar un color según el tipo
    colores = {'Mayor': 'darkred', 'Menor': 'green', 'Medio': 'royalblue'}
    color_grafica = colores.get(tipo_indice, 'black')

    # 4. Crear la gráfica (Plotly)
    fig = px.scatter(
        df_procesado,
        x='Fecha',
        y='Total_Victimas',
        title=f'Homicidios dolosos - {nombre_estado} (Índice {tipo_indice})',
        
        # 5. Datos para el 'hover' (al pasar el ratón)
        hover_data={
            'Año': True,
            'Mes': True,
            'Total_Victimas': True,
            'Fecha': False
        }
    )
    
    # 6. Personalizar la gráfica
    fig.update_layout(
        # font=dict(size=18),
        width=1100,
        height=1100,
        xaxis_title='Año',
        yaxis_title='Víctimas',
        yaxis_range=[0, df_procesado['Total_Victimas'].max() * 1.05],
        xaxis_gridcolor='rgba(0,0,0,0.1)',
        yaxis_gridcolor='rgba(0,0,0,0.1)',
        xaxis_gridwidth=1,
        yaxis_gridwidth=1,
        
        # --- TU REQUERIMIENTO ANTERIOR (INTERVALO DE 1 AÑO) ---
        xaxis_dtick="M12",      # Intervalo de 12 meses
        xaxis_tickformat="%Y"   # Mostrar solo el año
    )
    
    # 7. Actualizar color y opacidad de los puntos
    fig.update_traces(
        marker=dict(
            color=color_grafica, 
            size=6, 
            opacity=0.7
        )
    )
    
    # 8. Guardar la gráfica como HTML
    nombre_archivo = f'01-Limpieza Datos/01-b-Estatal/grafica_{nombre_hoja}.html'
    fig.write_html(nombre_archivo)
    print(f"--- Gráfica INTERACTIVA guardada como: '{nombre_archivo}'")

print("\nProceso completado.")