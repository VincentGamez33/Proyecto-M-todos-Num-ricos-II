import pandas as pd

# === 1. Definir rutas ===
input_file = '01-Limpieza Datos/resumen_homicidios_dolosos.xlsx'
output_file = '01-Limpieza Datos/01-b-Estatal/estatal_mensual_seleccion.xlsx'

# === 2. Cargar la hoja de datos correcta ===
try:
    df = pd.read_excel(input_file, sheet_name='Estatal_Mensual')
except Exception as e:
    print(f"Error al leer el archivo Excel: {e}")
    print("Asegúrate de que el archivo exista y contenga la hoja 'Estatal_Mensual'.")
    exit()

if "Entidad" not in df.columns:
    raise ValueError("No se encontró la columna 'Entidad' en la hoja 'Estatal_Mensual'.")
if "Total_Victimas" not in df.columns:
    raise ValueError("No se encontró la columna 'Total_Victimas' en la hoja 'Estatal_Mensual'.")

# === 3. Calcular la suma total de homicidios por estado ===
suma_total = df.groupby('Entidad')['Total_Victimas'].sum()

# === 4. Identificar los estados clave ===
estado_mayor = suma_total.idxmax()
estado_menor = suma_total.idxmin()
# .sort_values() ordena de menor a mayor por defecto
estado_promedio = suma_total.sort_values().index[len(suma_total)//2] # Esto da la mediana

print(f"Entidad con mayor índice (Total: {suma_total.max()}): {estado_mayor}")
print(f"Entidad con menor índice (Total: {suma_total.min()}): {estado_menor}")
print(f"Entidad con índice medio (Total: {suma_total[estado_promedio]}): {estado_promedio}")

# === 5. Filtrar los datos completos (mensuales) de cada estado ===
df_mayor = df[df["Entidad"] == estado_mayor]
df_menor = df[df["Entidad"] == estado_menor]
df_prom = df[df["Entidad"] == estado_promedio]

# === 6. Crear el archivo Excel final ===
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_mayor.to_excel(writer, sheet_name=f"Mayor_{estado_mayor}", index=False)
    df_menor.to_excel(writer, sheet_name=f"Menor_{estado_menor}", index=False)
    df_prom.to_excel(writer, sheet_name=f"Medio_{estado_promedio}", index=False)

print(f"Archivo '{output_file}' generado con éxito.")
