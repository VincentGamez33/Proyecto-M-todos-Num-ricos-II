import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
sys.path.append('02-Interpolacion')
sys.path.append('03-Splines')
sys.path.append('04-Diferenciacion')
sys.path.append('05-Integracion')

try:
    import interpolacion_lagrange as my_lagrange
    import interpolacion_splines as my_splines
    import diferenciacion_numerica as my_diff
    import integracion_numerica as my_int
    print("--- Módulos cargados ---")
except ImportError as e:
    print(f"[ERROR] Módulos no encontrados: {e}")
    exit()

CARPETA_DATOS = '01-Limpieza Datos/Archivos Limpios Excel'
patron_archivos = os.path.join(CARPETA_DATOS, "*.xlsx")
lista_archivos = glob.glob(patron_archivos)

if not lista_archivos:
    print("[ERROR] No hay archivos Excel.")
    exit()

# =========================================================
# 2. FUNCIÓN AUXILIAR: ENCONTRAR PICOS
# =========================================================
def encontrar_top_3_cambios(y_data):
    """
    Encuentra los índices de los 3 cambios más drásticos.
    """
    diferencias = np.abs(np.diff(y_data))
    candidatos = []
    
    for i in range(2, len(y_data) - 3):
        mag = diferencias[i] 
        candidatos.append((i, mag))
    
    candidatos.sort(key=lambda x: x[1], reverse=True)
    top_3_indices = [c[0] for c in candidatos[:3]]
    return sorted(top_3_indices)

# =========================================================
# 3. PROCESAMIENTO PRINCIPAL
# =========================================================
for ruta_archivo in lista_archivos:
    nombre_archivo = os.path.basename(ruta_archivo).replace(".xlsx", "") \
                                                    .replace("Mayor_", "") \
                                                    .replace("Medio_", "") \
                                                    .replace("Menor_", "")
    
    print(f"\n{'='*80}\nPROCESANDO ESTADO: {nombre_archivo.upper()}\n{'='*80}")

    try:
        df = pd.read_excel(ruta_archivo)
        meses_num = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
        df['Mes_Num'] = df['Mes'].map(meses_num)
        df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01')
        df = df.sort_values('Fecha')

        y_raw = df['Victimas'].values.astype(float)
        N_TOTAL = len(y_raw)
        x_norm = np.arange(1, N_TOTAL + 1) / N_TOTAL 

    except Exception as e:
        print(f"  [ERROR] Lectura: {e}")
        continue

    # --- PARÁMETROS ---
    N_PREDICT = 10     
    N_RUNGE = 25       
    BLOCK_SIZE = 5      

    x_full = x_norm
    y_full = y_raw
    x_train = x_norm[:-N_PREDICT]
    y_train = y_raw[:-N_PREDICT]
    x_test = x_norm[-N_PREDICT:] 
    y_test = y_raw[-N_PREDICT:]
    x_demo = x_train[:N_RUNGE]
    y_demo = y_train[:N_RUNGE]

    print("  -> Calculando curvas y modelos...")

    # 1. Lagrange Bloques
    x_lin_blocks_full = []
    y_lin_blocks_full = []
    idx = 0
    while idx <= len(x_full) - BLOCK_SIZE:
        xb = x_full[idx : idx + BLOCK_SIZE]
        yb = y_full[idx : idx + BLOCK_SIZE]
        xs = np.linspace(xb[0], xb[-1], 25)
        ys = [my_lagrange.lagrange_polynomials(xb, yb, z) for z in xs]
        x_lin_blocks_full.extend(xs)
        y_lin_blocks_full.extend(ys)
        idx += (BLOCK_SIZE - 1)
    if idx < len(x_full) - 1:
        xb = x_full[idx:]
        yb = y_full[idx:]
        if len(xb) >= 2:
            xs = np.linspace(xb[0], xb[-1], 10)
            ys = [my_lagrange.lagrange_polynomials(xb, yb, z) for z in xs]
            x_lin_blocks_full.extend(xs)
            y_lin_blocks_full.extend(ys)

    # 2. Splines Cúbicos
    x_lin_spline_full = []
    y_lin_spline_full = []
    idx = 0
    while idx <= len(x_full) - BLOCK_SIZE:
        xb = x_full[idx : idx + BLOCK_SIZE]
        yb = y_full[idx : idx + BLOCK_SIZE]
        sp = my_splines.spline_cubico_natural(xb, yb)
        xs = np.linspace(xb[0], xb[-1], 25)
        ys = sp(xs)
        x_lin_spline_full.extend(xs)
        y_lin_spline_full.extend(ys)
        idx += (BLOCK_SIZE - 1)
    if idx < len(x_full) - 1:
        xb = x_full[idx:]
        yb = y_full[idx:]
        if len(xb) >= 2:
            sp = my_splines.spline_cubico_natural(xb, yb)
            xs = np.linspace(xb[0], xb[-1], 10)
            ys = sp(xs)
            x_lin_spline_full.extend(xs)
            y_lin_spline_full.extend(ys)

    # Predicción Futura
    xb_last_train = x_train[-BLOCK_SIZE:]
    yb_last_train = y_train[-BLOCK_SIZE:]
    last_spline_func = my_splines.spline_cubico_natural(xb_last_train, yb_last_train)
    y_pred_spline = last_spline_func(x_test)
    y_pred_lagrange = np.array([my_lagrange.lagrange_polynomials(xb_last_train, yb_last_train, val) for val in x_test])

    # --- GRAFICACIÓN ESTÁNDAR ---
    print("  -> Generando gráficas estándar...")
    def guardar(subfijo, titulo, func_plot):
        plt.figure(figsize=(12, 6))
        func_plot()
        plt.title(f"{titulo} ({nombre_archivo})")
        plt.xlabel(f"Tiempo (Total: {N_TOTAL} meses)")
        plt.ylabel("Víctimas")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"Grafica_{subfijo}_{nombre_archivo}.pdf")
        plt.close()

    x_lin_runge = np.linspace(x_demo[0], x_demo[-1], 200)
    try: y_lin_runge = [my_lagrange.lagrange_polynomials(x_demo, y_demo, z) for z in x_lin_runge]
    except: y_lin_runge = np.zeros_like(x_lin_runge)

    # 1. Runge
    guardar("1_Runge", "Lagrange Global (Fenómeno de Runge)", lambda: (
        plt.scatter(x_demo, y_demo, c='k', label='Datos'),
        plt.plot(x_lin_runge, y_lin_runge, c='r', label='Lagrange Global'),
        plt.ylim(min(y_demo)*0.8, max(y_demo)*1.2)
    ))

    # 2. Lagrange Bloques
    guardar("2_Lagrange_Bloques", "Lagrange por Bloques", lambda: (
        plt.scatter(x_full, y_full, c='k', s=10, label='Todos los Datos'),
        plt.plot(x_lin_blocks_full, y_lin_blocks_full, c='g', linewidth=1, label='Lagrange Bloques')
    ))

    # 3. Comparación
    guardar("3_Lagrange_comparacion", "Comparación Lagrange (Global vs Bloques)", lambda: (
        plt.scatter(x_demo, y_demo, c='k', alpha=0.3, label='Datos (Tramo Global)'),
        plt.plot(x_lin_runge, y_lin_runge, c='r', linestyle='--', alpha=0.6, label='Global (Runge)'),
        plt.plot(x_lin_blocks_full, y_lin_blocks_full, c='g', linewidth=1.5, label='Por Bloques'),
        plt.ylim(min(y_full)*0.8, max(y_full)*1.2)
    ))

    # 4. Splines
    guardar("4_Splines", "Splines Cúbicos", lambda: (
        plt.scatter(x_full, y_full, c='k', s=10, label='Todos los Datos'),
        plt.plot(x_lin_spline_full, y_lin_spline_full, c='b', linewidth=1.5, label='Splines Cúbicos')
    ))

    # 5. Predicción
    guardar("5_Prediccion", "Estimación Futura", lambda: (
        plt.scatter(x_train[-20:], y_train[-20:], c='k', label='Historia Reciente'),
        plt.scatter(x_test, y_test, c='g', marker='*', s=180, label='Realidad', zorder=10),
        plt.plot(x_test, y_pred_spline, c='purple', linestyle='--', linewidth=2, label='Splines'),
        plt.plot(x_test, y_pred_lagrange, c='orange', linestyle='-.', linewidth=2, label='Lagrange')
    ))

    # ===================================================================
    # ANÁLISIS DE PICOS (CAMBIOS DRÁSTICOS)
    # ===================================================================
    print("  -> Generando análisis de picos drásticos...")

    indices_criticos = encontrar_top_3_cambios(y_raw)
    print(f"     Picos detectados en índices: {indices_criticos}")

    # --- GRÁFICA 6: PICOS ---
    plt.figure(figsize=(12, 6))
    plt.scatter(x_full, y_full, c='k', s=15, label='Datos Reales')
    plt.plot(x_lin_spline_full, y_lin_spline_full, c='b', alpha=0.5, label='Splines')

    for i, idx_centro in enumerate(indices_criticos):
        idx_inicio = idx_centro - 2
        idx_fin = idx_centro + 3 
        xb_local = x_norm[idx_inicio : idx_fin]
        yb_local = y_raw[idx_inicio : idx_fin]
        fecha_evento = df['Fecha'].iloc[idx_centro].strftime('%Y-%B')
        
        plt.scatter(xb_local[2], yb_local[2], c='red', s=150, marker='X', zorder=10, 
                    label=f'Pico {i+1}' if i==0 else "")
        
        sp_local = my_splines.spline_cubico_natural(xb_local, yb_local)
        punto_eval = xb_local[2]
        deriv_ref = my_diff.central_der_5steps(sp_local, 1e-9, punto_eval)
        
        print("\n" + "-"*90)
        print(f"  ANÁLISIS DE PICO #{i+1}: {fecha_evento} (Índice {idx_centro})")
        print("-" * 90)
        print(f"  * Bloque: {yb_local}")
        print(f"  * Tendencia Real (f'): {deriv_ref:.4f}")
        print("-" * 90)
        print(f"  {'h (Paso)':<12} | {'Derivada Calc':<15} | {'E. Abs (en)':<15} | {'E. Norm (C)':<18} | {'E. Rel (er)':<15}")
        print("-" * 90)
        
        h_real = 1.0/N_TOTAL
        pasos_prueba = [h_real*4, h_real*2, h_real, h_real/2, h_real/10]
        
        for h in pasos_prueba:
            d_calc = my_diff.central_der_5steps(sp_local, h, punto_eval)
            e_n, e_norm, e_rel = my_diff.calcular_errores(deriv_ref, d_calc, h, orden_metodo=4)
            print(f"  {h:<12.6f} | {d_calc:<15.6f} | {e_n:<15.8f} | {e_norm:<18.6f} | {e_rel:<15.6f}")

    plt.title(f"Puntos de Cambio Drástico (Picos) - {nombre_archivo}")
    plt.xlabel("Tiempo Normalizado")
    plt.ylabel("Víctimas")
    plt.legend()
    plt.savefig(f"Grafica_6_Puntos_criticos_{nombre_archivo}.pdf")
    plt.close()
    
    # --- GRÁFICA 7: DERIVADA GLOBAL ---
    print("  -> Generando gráfica de Derivada Global...")
    sp_global = my_splines.spline_cubico_natural(x_full, y_full)
    x_dense = np.linspace(x_full[0], x_full[-1], 500)
    y_dense = sp_global(x_dense)
    dy_dense = np.gradient(y_dense, x_dense)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_dense, dy_dense, c='darkred', linewidth=1.5, label="f '(x) - Velocidad de cambio")
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8) 
    plt.title(f"Perfil de Velocidad (Derivada) - {nombre_archivo}")
    plt.xlabel("Tiempo Normalizado")
    plt.ylabel("Tasa de Cambio (Víctimas/Tiempo)")
    plt.fill_between(x_dense, dy_dense, 0, where=(dy_dense>0), color='red', alpha=0.1, label="Incremento")
    plt.fill_between(x_dense, dy_dense, 0, where=(dy_dense<0), color='green', alpha=0.1, label="Decremento")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Grafica_7_Derivada_Total_{nombre_archivo}.pdf")
    plt.close()

    # ===================================================================
    # ANÁLISIS DE INTEGRACIÓN NUMÉRICA (USANDO TUS FUNCIONES)
    # ===================================================================
    print("  -> Generando análisis de integración (Comparativo)...")
    
    a_int = x_full[0]
    b_int = x_full[-1]
    n_int = 100 
    
    # 1. Calcular escalares totales usando TUS funciones
    val_riemann = my_int.integral_riemann_uniform(sp_global, a_int, b_int, n_int)
    val_trap = my_int.integral_trapezoidal_rule_uniform(sp_global, a_int, b_int, n_int)
    val_simpson = my_int.integral_simpson_rule_uniform(sp_global, a_int, b_int, n_int)
    
    # Calcular diferencias relativas (usando Simpson como "Verdad")
    err_riemann = abs(val_riemann - val_simpson)
    err_trap = abs(val_trap - val_simpson)
    
    print("\n" + "="*80)
    print(f"  TABLA DE INTEGRACIÓN NUMÉRICA (Área Total) - {nombre_archivo}")
    print("="*80)
    print(f"  {'MÉTODO':<25} | {'RESULTADO':<20} | {'DIFERENCIA ABS':<20}")
    print("-" * 80)
    print(f"  {'Riemann':<25} | {val_riemann:<20.6f} | {err_riemann:<20.6f}")
    print(f"  {'Trapecio':<25} | {val_trap:<20.6f} | {err_trap:<20.6f}")
    print(f"  {'Simpson 1/3':<25} | {val_simpson:<20.6f} | {'(Referencia)':<20}")
    print("="*80 + "\n")
    
    # --- GRÁFICA 8: INTEGRAL ACUMULADA (Visualización del Fenómeno) ---
    print("  -> Generando Gráfica 8 (Acumulación)...")
    
    y_integral_acumulada = []
    # Calculamos la acumulada suave usando Trapecio para la forma de la curva
    for i, xi in enumerate(x_dense):
        if i == 0: y_integral_acumulada.append(0.0)
        else:
            val = my_int.integral_trapezoidal_rule_uniform(sp_global, a_int, xi, i)
            y_integral_acumulada.append(val)
            
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_dense, y_dense, c='b', label='f(x) - Víctimas')
    plt.fill_between(x_dense, y_dense, alpha=0.2, color='b')
    plt.title(f"Función de Densidad")
    plt.xlabel("Tiempo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_dense, y_integral_acumulada, c='purple', linewidth=2, label='F(x)')
    plt.title(f"Acumulación Total ~ {val_simpson:.0f} víctimas")
    plt.xlabel("Tiempo")
    plt.ylabel("Total Acumulado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"Dinámica de Acumulación - {nombre_archivo}")
    plt.savefig(f"Grafica_8_Integral_Acumulada_{nombre_archivo}.pdf")
    plt.close()

    # --- GRÁFICA 9: COMPARACIÓN DE MÉTODOS (La nueva gráfica prudente) ---
    print("  -> Generando Gráfica 9 (Comparación de Métodos)...")
    
    metodos = ['Riemann', 'Trapecio', 'Simpson 1/3']
    valores = [val_riemann, val_trap, val_simpson]
    colores = ['#FF9999', '#66B2FF', '#99FF99'] # Rojo suave, Azul suave, Verde suave
    
    plt.figure(figsize=(10, 6))
    
    # Crear gráfico de barras
    barras = plt.bar(metodos, valores, color=colores, edgecolor='black', alpha=0.7)
    
    # Línea de referencia (Simpson)
    plt.axhline(y=val_simpson, color='green', linestyle='--', linewidth=1, label='Ref. Simpson')
    
    # Ajustar el eje Y para que se noten las diferencias (hacer "zoom" en la cima)
    min_val = min(valores) * 0.99
    max_val = max(valores) * 1.01
    plt.ylim(min_val, max_val)
    
    # Etiquetas con los valores exactos sobre las barras
    for bar, val in zip(barras, valores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.2f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.title(f"Comparación de Métodos de Integración Numérica - {nombre_archivo}")
    plt.ylabel("Total de Víctimas Estimado")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(f"Grafica_9_Comparacion_Integrales_{nombre_archivo}.pdf")
    plt.close()

    print("="*90)

print("\n¡PROCESO COMPLETADO EXITOSAMENTE!")