import pandas as pd
import duckdb

print("--- Iniciando proceso ---")

# --- ETAPA 1: PANDAS (Preparación) ---
print("Cargando y limpiando datos con Pandas...")
df = pd.read_csv('01-Limpieza Datos/Estatal-Víctimas-2015-2025_sep2025.csv', encoding='latin1')
df_hd = df[df['Subtipo de delito'] == 'Homicidio doloso'].copy()

month_columns = [
    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
]
id_vars = [col for col in df_hd.columns if col not in month_columns]

# Transformamos (melt) primero
df_tidy = df_hd.melt(
    id_vars=id_vars,
    value_vars=month_columns,
    var_name='Mes',
    value_name='Victimas'
)
print(f"DataFrame 'df_tidy' (pre-filtro) listo con {len(df_tidy)} filas.")

# Filtramos los NaN (meses futuros de 2025)
df_tidy = df_tidy.dropna(subset=['Victimas'])
print(f"DataFrame 'df_tidy' (post-filtro) tiene {len(df_tidy)} filas.")

# Convertimos a entero
df_tidy['Victimas'] = df_tidy['Victimas'].astype(int)


# --- ETAPA 2: DUCKDB (Consulta SQL) ---
print("Consultando DataFrame 'df_tidy' con DuckDB...")

# --- Consulta 1: Nacional Mensual ---
query_nacional = """
SELECT 
    Año, 
    Mes, 
    SUM(Victimas) AS Total_Victimas
FROM df_tidy
GROUP BY Año, Mes
ORDER BY 
    Año, 
    CASE Mes
        WHEN 'Enero' THEN 1 WHEN 'Febrero' THEN 2 WHEN 'Marzo' THEN 3
        WHEN 'Abril' THEN 4 WHEN 'Mayo' THEN 5 WHEN 'Junio' THEN 6
        WHEN 'Julio' THEN 7 WHEN 'Agosto' THEN 8 WHEN 'Septiembre' THEN 9
        WHEN 'Octubre' THEN 10 WHEN 'Noviembre' THEN 11 WHEN 'Diciembre' THEN 12
    END
"""
resumen_nacional_db = duckdb.sql(query_nacional).to_df()
print("Resumen 1: Nacional Mensual generado.")

# --- Consulta 2: Estatal Mensual ---
query_estatal = """
SELECT 
    Entidad, 
    Año, 
    Mes, 
    SUM(Victimas) AS Total_Victimas
FROM df_tidy
GROUP BY Entidad, Año, Mes
ORDER BY 
    Entidad, 
    Año, 
    CASE Mes
        WHEN 'Enero' THEN 1 WHEN 'Febrero' THEN 2 WHEN 'Marzo' THEN 3
        WHEN 'Abril' THEN 4 WHEN 'Mayo' THEN 5 WHEN 'Junio' THEN 6
        WHEN 'Julio' THEN 7 WHEN 'Agosto' THEN 8 WHEN 'Septiembre' THEN 9
        WHEN 'Octubre' THEN 10 WHEN 'Noviembre' THEN 11 WHEN 'Diciembre' THEN 12
    END
"""
resumen_estatal_db = duckdb.sql(query_estatal).to_df()
print("Resumen 2: Estatal Mensual generado.")

# --- Consulta 3: Estatal Anual ---
query_estatal_anual = """
SELECT 
    Entidad, 
    Año, 
    SUM(Victimas) AS Total_Victimas_Anual
FROM df_tidy
GROUP BY Entidad, Año
ORDER BY Entidad, Año
"""
resumen_estatal_anual_db = duckdb.sql(query_estatal_anual).to_df()
print(f"Resumen 3: Estatal Anual generado con {len(resumen_estatal_anual_db)} filas.")

# --- *NUEVA* Consulta 4: Nacional Anual por Modalidad ---
query_nacional_anual_modalidad = """
SELECT 
    Año,
    Modalidad,
    SUM(Victimas) AS Total_Victimas
FROM df_tidy
GROUP BY Año, Modalidad
ORDER BY Año, Modalidad
"""
resumen_nacional_modalidad_db = duckdb.sql(query_nacional_anual_modalidad).to_df()
print(f"Resumen 4: Nacional Anual por Modalidad generado con {len(resumen_nacional_modalidad_db)} filas.")

# --- *NUEVA* Consulta 5: Estatal Anual por Modalidad ---
query_estatal_anual_modalidad = """
SELECT 
    Entidad,
    Año,
    Modalidad,
    SUM(Victimas) AS Total_Victimas
FROM df_tidy
GROUP BY Entidad, Año, Modalidad
ORDER BY Entidad, Año, Modalidad
"""
resumen_estatal_modalidad_db = duckdb.sql(query_estatal_anual_modalidad).to_df()
print(f"Resumen 5: Estatal Anual por Modalidad generado con {len(resumen_estatal_modalidad_db)} filas.")


# --- ETAPA 3: OPENPYXL / PANDAS (Exportación) ---
output_excel_file = '01-Limpieza Datos/resumen_homicidios_dolosos.xlsx'
print(f"Guardando 5 resúmenes en '{output_excel_file}'...")

with pd.ExcelWriter(output_excel_file) as writer:
    resumen_nacional_db.to_excel(writer, sheet_name='Nacional_Mensual', index=False)
    resumen_estatal_db.to_excel(writer, sheet_name='Estatal_Mensual', index=False)
    resumen_estatal_anual_db.to_excel(writer, sheet_name='Estatal_Anual', index=False)
    # --- Añadimos las dos nuevas hojas ---
    resumen_nacional_modalidad_db.to_excel(writer, sheet_name='Nacional_Anual_Modalidad', index=False)
    resumen_estatal_modalidad_db.to_excel(writer, sheet_name='Estatal_Anual_Modalidad', index=False)

print("¡Proceso completado con éxito! El acrhivo Excel ahora tiene 5 hojas.")
