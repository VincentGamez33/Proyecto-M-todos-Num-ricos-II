# import pandas as pd
# import duckdb

# print("--- Iniciando proceso ---")

# # --- ETAPA 1: PANDAS (Preparación) ---
# print("Cargando y limpiando datos con Pandas...")
# df = pd.read_csv('01-Limpieza Datos/Estatal-Víctimas-2015-2025_sep2025.csv', encoding='latin1')
# df_hd = df[df['Subtipo de delito'] == 'Homicidio doloso'].copy()

# month_columns = [
#     'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
#     'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
# ]
# id_vars = [col for col in df_hd.columns if col not in month_columns]

# # Transformamos (melt) primero
# df_tidy = df_hd.melt(
#     id_vars=id_vars,
#     value_vars=month_columns,
#     var_name='Mes',
#     value_name='Victimas'
# )
# print(f"DataFrame 'df_tidy' (pre-filtro) listo con {len(df_tidy)} filas.")

# # Filtramos los NaN (meses futuros de 2025)
# df_tidy = df_tidy.dropna(subset=['Victimas'])
# print(f"DataFrame 'df_tidy' (post-filtro) tiene {len(df_tidy)} filas.")

# # Convertimos a entero
# df_tidy['Victimas'] = df_tidy['Victimas'].astype(int)


# # --- ETAPA 2: DUCKDB (Consulta SQL) ---
# print("Consultando DataFrame 'df_tidy' con DuckDB...")

# # --- Consulta 1: Nacional Mensual ---
# query_nacional = """
# SELECT 
#     Año, 
#     Mes, 
#     SUM(Victimas) AS Total_Victimas
# FROM df_tidy
# GROUP BY Año, Mes
# ORDER BY 
#     Año, 
#     CASE Mes
#         WHEN 'Enero' THEN 1 WHEN 'Febrero' THEN 2 WHEN 'Marzo' THEN 3
#         WHEN 'Abril' THEN 4 WHEN 'Mayo' THEN 5 WHEN 'Junio' THEN 6
#         WHEN 'Julio' THEN 7 WHEN 'Agosto' THEN 8 WHEN 'Septiembre' THEN 9
#         WHEN 'Octubre' THEN 10 WHEN 'Noviembre' THEN 11 WHEN 'Diciembre' THEN 12
#     END
# """
# resumen_nacional_db = duckdb.sql(query_nacional).to_df()
# print("Resumen 1: Nacional Mensual generado.")

# # --- Consulta 2: Estatal Mensual ---
# query_estatal = """
# SELECT 
#     Entidad, 
#     Año, 
#     Mes, 
#     SUM(Victimas) AS Total_Victimas
# FROM df_tidy
# GROUP BY Entidad, Año, Mes
# ORDER BY 
#     Entidad, 
#     Año, 
#     CASE Mes
#         WHEN 'Enero' THEN 1 WHEN 'Febrero' THEN 2 WHEN 'Marzo' THEN 3
#         WHEN 'Abril' THEN 4 WHEN 'Mayo' THEN 5 WHEN 'Junio' THEN 6
#         WHEN 'Julio' THEN 7 WHEN 'Agosto' THEN 8 WHEN 'Septiembre' THEN 9
#         WHEN 'Octubre' THEN 10 WHEN 'Noviembre' THEN 11 WHEN 'Diciembre' THEN 12
#     END
# """
# resumen_estatal_db = duckdb.sql(query_estatal).to_df()
# print("Resumen 2: Estatal Mensual generado.")

# # --- Consulta 3: Estatal Anual ---
# query_estatal_anual = """
# SELECT 
#     Entidad, 
#     Año, 
#     SUM(Victimas) AS Total_Victimas_Anual
# FROM df_tidy
# GROUP BY Entidad, Año
# ORDER BY Entidad, Año
# """
# resumen_estatal_anual_db = duckdb.sql(query_estatal_anual).to_df()
# print(f"Resumen 3: Estatal Anual generado con {len(resumen_estatal_anual_db)} filas.")

# # --- *NUEVA* Consulta 4: Nacional Anual por Modalidad ---
# query_nacional_anual_modalidad = """
# SELECT 
#     Año,
#     Modalidad,
#     SUM(Victimas) AS Total_Victimas
# FROM df_tidy
# GROUP BY Año, Modalidad
# ORDER BY Año, Modalidad
# """
# resumen_nacional_modalidad_db = duckdb.sql(query_nacional_anual_modalidad).to_df()
# print(f"Resumen 4: Nacional Anual por Modalidad generado con {len(resumen_nacional_modalidad_db)} filas.")

# # --- *NUEVA* Consulta 5: Estatal Anual por Modalidad ---
# query_estatal_anual_modalidad = """
# SELECT 
#     Entidad,
#     Año,
#     Modalidad,
#     SUM(Victimas) AS Total_Victimas
# FROM df_tidy
# GROUP BY Entidad, Año, Modalidad
# ORDER BY Entidad, Año, Modalidad
# """
# resumen_estatal_modalidad_db = duckdb.sql(query_estatal_anual_modalidad).to_df()
# print(f"Resumen 5: Estatal Anual por Modalidad generado con {len(resumen_estatal_modalidad_db)} filas.")


# # --- ETAPA 3: OPENPYXL / PANDAS (Exportación) ---
# output_excel_file = '01-Limpieza Datos/resumen_homicidios_dolosos.xlsx'
# print(f"Guardando 5 resúmenes en '{output_excel_file}'...")

# with pd.ExcelWriter(output_excel_file) as writer:
#     resumen_nacional_db.to_excel(writer, sheet_name='Nacional_Mensual', index=False)
#     resumen_estatal_db.to_excel(writer, sheet_name='Estatal_Mensual', index=False)
#     resumen_estatal_anual_db.to_excel(writer, sheet_name='Estatal_Anual', index=False)
#     # --- Añadimos las dos nuevas hojas ---
#     resumen_nacional_modalidad_db.to_excel(writer, sheet_name='Nacional_Anual_Modalidad', index=False)
#     resumen_estatal_modalidad_db.to_excel(writer, sheet_name='Estatal_Anual_Modalidad', index=False)

# print("¡Proceso completado con éxito! El archivo Excel ahora tiene 5 hojas.")


import pandas as pd
import numpy as np
import os

# --- CONFIGURACIÓN ---
INPUT_FILE = '01-Limpieza Datos/Estatal-Víctimas-2015-2025_sep2025.csv'
OUTPUT_FOLDER = '01-Limpieza Datos/Archivos Limpios Excel'

# Crear carpeta de salida si no existe
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Carpeta creada: {OUTPUT_FOLDER}")

print("--- 1. Cargando y limpiando datos crudos ---")
try:
    df = pd.read_csv(INPUT_FILE, encoding='latin1')
except FileNotFoundError:
    print(f"Error: No se encuentra {INPUT_FILE}")
    exit()

# Filtrar y Transformar (Melt)
df = df[df['Subtipo de delito'] == 'Homicidio doloso'].copy()

month_columns = [
    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
]
id_vars = [col for col in df.columns if col not in month_columns]

df_melt = df.melt(id_vars=id_vars, value_vars=month_columns, var_name='Mes', value_name='Victimas')
df_melt = df_melt.dropna(subset=['Victimas'])
df_melt['Victimas'] = df_melt['Victimas'].astype(int)

# Mapa numérico para ordenar correctamente
mes_map = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}
df_melt['Mes_Num'] = df_melt['Mes'].map(mes_map)

# =========================================================
# 2. SELECCIÓN DE ESTADOS (Mayor, Medio, Menor)
# =========================================================
print("--- 2. Calculando ranking... ---")
ranking = df_melt.groupby('Entidad')['Victimas'].sum().sort_values(ascending=False)

estado_mayor = ranking.index[0]
estado_menor = ranking.index[-1]
estado_medio = ranking.index[len(ranking)//2]

print(f" -> Top (Mayor): {estado_mayor}")
print(f" -> Medio:       {estado_medio}")
print(f" -> Bottom (Menor): {estado_menor}")

# =========================================================
# 3. EXPORTACIÓN A ARCHIVOS SEPARADOS
# =========================================================
print(f"--- 3. Guardando Excels en '{OUTPUT_FOLDER}' ---")

def guardar_excel(nombre_archivo, entidad=None):
    """Filtra y guarda un Excel individual"""
    if entidad:
        data = df_melt[df_melt['Entidad'] == entidad].copy()
        # Mantenemos columna Entidad para referencia
        cols = ['Entidad', 'Año', 'Mes', 'Victimas'] 
    else:
        # Para Nacional agrupamos todo
        data = df_melt.copy()
        cols = ['Año', 'Mes', 'Victimas']

    # Agrupar cronológicamente
    grouped = data.groupby(['Año', 'Mes_Num', 'Mes'])['Victimas'].sum().reset_index()
    grouped = grouped.sort_values(by=['Año', 'Mes_Num'])
    
    # Si es estatal, re-agregamos el nombre del estado para que el Excel sea claro
    if entidad:
        grouped['Entidad'] = entidad
        final_df = grouped[cols]
    else:
        final_df = grouped[cols]

    # Guardar
    ruta = os.path.join(OUTPUT_FOLDER, f"{nombre_archivo}.xlsx")
    final_df.to_excel(ruta, index=False)
    print(f" -> Generado: {nombre_archivo}.xlsx")

# Generamos los 4 archivos
guardar_excel("Nacional", None)
guardar_excel(f"{estado_mayor}", estado_mayor)
guardar_excel(f"{estado_medio}", estado_medio)
guardar_excel(f"{estado_menor}", estado_menor)

print("¡Listo! Proceso terminado.")
