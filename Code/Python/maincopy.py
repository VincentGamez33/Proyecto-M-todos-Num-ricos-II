# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import os
# import glob

# # =========================================================
# # 1. CONFIGURACIÓN
# # =========================================================
# sys.path.append('02-Interpolacion')
# sys.path.append('03-Splines')
# sys.path.append('04-Diferenciacion')
# sys.path.append('05-Integracion')

# try:
#     import interpolacion_lagrange as my_lagrange
#     import interpolacion_splines as my_splines
#     import diferenciacion_numerica as my_diff
#     import integracion_numerica as my_int
#     print("--- Módulos cargados ---")
# except ImportError as e:
#     print(f"[ERROR] Módulos no encontrados: {e}")
#     exit()

# CARPETA_DATOS = '01-Limpieza Datos/Archivos Limpios Excel'
# patron_archivos = os.path.join(CARPETA_DATOS, "*.xlsx")
# lista_archivos = glob.glob(patron_archivos)

# if not lista_archivos:
#     print("[ERROR] No hay archivos Excel.")
#     exit()

# # =========================================================
# # 2. FUNCIÓN AUXILIAR: ENCONTRAR PICOS
# # =========================================================
# def encontrar_top_3_cambios(y_data):
#     """
#     Encuentra los índices de los 3 cambios más drásticos (mayor pendiente).
#     Ignora los primeros 2 y últimos 2 puntos para permitir bloques de 5.
#     Retorna: Lista de 3 índices ordenados cronológicamente.
#     """
#     diferencias = np.abs(np.diff(y_data))
#     candidatos = []
    
#     # Rango seguro: desde índice 2 hasta N-3 para tener vecinos suficientes
#     for i in range(2, len(y_data) - 3):
#         mag = diferencias[i] 
#         candidatos.append((i, mag))
    
#     # Ordenar por magnitud del cambio (de mayor a menor)
#     candidatos.sort(key=lambda x: x[1], reverse=True)
    
#     # Devolver los índices de los top 3
#     top_3_indices = [c[0] for c in candidatos[:3]]
    
#     return sorted(top_3_indices)

# # =========================================================
# # 3. PROCESAMIENTO PRINCIPAL
# # =========================================================
# for ruta_archivo in lista_archivos:
#     nombre_archivo = os.path.basename(ruta_archivo).replace(".xlsx", "") \
#                                                    .replace("Mayor_", "") \
#                                                    .replace("Medio_", "") \
#                                                    .replace("Menor_", "")
    
#     print(f"\n{'='*80}\nPROCESANDO ESTADO: {nombre_archivo.upper()}\n{'='*80}")

#     try:
#         df = pd.read_excel(ruta_archivo)
#         meses_num = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
#         df['Mes_Num'] = df['Mes'].map(meses_num)
#         df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01')
#         df = df.sort_values('Fecha')

#         y_raw = df['Victimas'].values.astype(float)
#         N_TOTAL = len(y_raw)
#         x_norm = np.arange(1, N_TOTAL + 1) / N_TOTAL 

#     except Exception as e:
#         print(f"  [ERROR] Lectura: {e}")
#         continue

#     # --- PARÁMETROS ---
#     N_PREDICT = 10     
#     N_RUNGE = 25       
#     BLOCK_SIZE = 5      

#     x_full = x_norm
#     y_full = y_raw
#     x_train = x_norm[:-N_PREDICT]
#     y_train = y_raw[:-N_PREDICT]
#     x_test = x_norm[-N_PREDICT:] 
#     y_test = y_raw[-N_PREDICT:]
#     x_demo = x_train[:N_RUNGE]
#     y_demo = y_train[:N_RUNGE]

#     print("  -> Calculando curvas y modelos...")

#     # 1. Lagrange Bloques (Completo)
#     x_lin_blocks_full = []
#     y_lin_blocks_full = []
#     idx = 0
#     while idx <= len(x_full) - BLOCK_SIZE:
#         xb = x_full[idx : idx + BLOCK_SIZE]
#         yb = y_full[idx : idx + BLOCK_SIZE]
#         xs = np.linspace(xb[0], xb[-1], 25)
#         ys = [my_lagrange.lagrange_polynomials(xb, yb, z) for z in xs]
#         x_lin_blocks_full.extend(xs)
#         y_lin_blocks_full.extend(ys)
#         idx += (BLOCK_SIZE - 1)
#     if idx < len(x_full) - 1:
#         xb = x_full[idx:]
#         yb = y_full[idx:]
#         if len(xb) >= 2:
#             xs = np.linspace(xb[0], xb[-1], 10)
#             ys = [my_lagrange.lagrange_polynomials(xb, yb, z) for z in xs]
#             x_lin_blocks_full.extend(xs)
#             y_lin_blocks_full.extend(ys)

#     # 2. Splines Cúbicos (Completo)
#     x_lin_spline_full = []
#     y_lin_spline_full = []
#     idx = 0
#     while idx <= len(x_full) - BLOCK_SIZE:
#         xb = x_full[idx : idx + BLOCK_SIZE]
#         yb = y_full[idx : idx + BLOCK_SIZE]
#         sp = my_splines.spline_cubico_natural(xb, yb)
#         xs = np.linspace(xb[0], xb[-1], 25)
#         ys = sp(xs)
#         x_lin_spline_full.extend(xs)
#         y_lin_spline_full.extend(ys)
#         idx += (BLOCK_SIZE - 1)
#     if idx < len(x_full) - 1:
#         xb = x_full[idx:]
#         yb = y_full[idx:]
#         if len(xb) >= 2:
#             sp = my_splines.spline_cubico_natural(xb, yb)
#             xs = np.linspace(xb[0], xb[-1], 10)
#             ys = sp(xs)
#             x_lin_spline_full.extend(xs)
#             y_lin_spline_full.extend(ys)

#     # Predicción Futura
#     xb_last_train = x_train[-BLOCK_SIZE:]
#     yb_last_train = y_train[-BLOCK_SIZE:]
#     last_spline_func = my_splines.spline_cubico_natural(xb_last_train, yb_last_train)
#     y_pred_spline = last_spline_func(x_test)
#     y_pred_lagrange = np.array([my_lagrange.lagrange_polynomials(xb_last_train, yb_last_train, val) for val in x_test])

#     # --- GRAFICACIÓN ESTÁNDAR (Las 4 gráficas de siempre) ---
#     print("  -> Generando gráficas estándar...")
#     def guardar(subfijo, titulo, func_plot):
#         plt.figure(figsize=(12, 6))
#         func_plot()
#         plt.title(f"{titulo} ({nombre_archivo})")
#         plt.xlabel(f"Tiempo (Total: {N_TOTAL} meses)")
#         plt.ylabel("Víctimas")
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(f"Grafica_{subfijo}_{nombre_archivo}.pdf")
#         plt.close()

#     x_lin_runge = np.linspace(x_demo[0], x_demo[-1], 200)
#     try: y_lin_runge = [my_lagrange.lagrange_polynomials(x_demo, y_demo, z) for z in x_lin_runge]
#     except: y_lin_runge = np.zeros_like(x_lin_runge)

#     guardar("1_Runge", "1. Fenómeno de Runge", lambda: (
#         plt.scatter(x_demo, y_demo, c='k', label='Datos'),
#         plt.plot(x_lin_runge, y_lin_runge, c='r', label='Lagrange Global'),
#         plt.ylim(min(y_demo)*0.8, max(y_demo)*1.2)
#     ))
#     guardar("2_Lagrange_Bloques", "2. Lagrange Bloques", lambda: (
#         plt.scatter(x_full, y_full, c='k', s=10, label='Todos los Datos'),
#         plt.plot(x_lin_blocks_full, y_lin_blocks_full, c='g', linewidth=1, label='Lagrange Bloques')
#     ))
#     guardar("3_Splines", "3. Splines Cúbicos", lambda: (
#         plt.scatter(x_full, y_full, c='k', s=10, label='Todos los Datos'),
#         plt.plot(x_lin_spline_full, y_lin_spline_full, c='b', linewidth=1.5, label='Splines Cúbicos')
#     ))
#     guardar("4_Prediccion", "4. Estimación Futura", lambda: (
#         plt.scatter(x_train[-20:], y_train[-20:], c='k', label='Historia Reciente'),
#         plt.scatter(x_test, y_test, c='g', marker='*', s=180, label='Realidad', zorder=10),
#         plt.plot(x_test, y_pred_spline, c='purple', linestyle='--', linewidth=2, label='Splines'),
#         plt.plot(x_test, y_pred_lagrange, c='orange', linestyle='-.', linewidth=2, label='Lagrange')
#     ))

#     # ===================================================================
#     # ANÁLISIS DE PICOS (CAMBIOS DRÁSTICOS) Y TABLAS DE CONVERGENCIA
#     # ===================================================================
#     print("  -> Generando análisis de picos drásticos...")

#     # 1. Encontrar los picos
#     indices_criticos = encontrar_top_3_cambios(y_raw)
#     print(f"     Picos detectados en índices: {indices_criticos}")

#     # 2. Configurar la Gráfica de Picos (Fondo: Datos + Splines Suaves)
#     plt.figure(figsize=(12, 6))
#     plt.scatter(x_full, y_full, c='k', s=15, label='Datos Reales')
#     plt.plot(x_lin_spline_full, y_lin_spline_full, c='b', alpha=0.5, label='Splines')

#     # 3. Iterar sobre los 3 picos para analizar y marcar
#     for i, idx_centro in enumerate(indices_criticos):
        
#         # A) Definir bloque local
#         idx_inicio = idx_centro - 2
#         idx_fin = idx_centro + 3 
#         xb_local = x_norm[idx_inicio : idx_fin]
#         yb_local = y_raw[idx_inicio : idx_fin]
        
#         fecha_evento = df['Fecha'].iloc[idx_centro].strftime('%Y-%B')
        
#         # B) Marcar el pico en la gráfica con una X roja grande
#         plt.scatter(xb_local[2], yb_local[2], c='red', s=150, marker='X', zorder=10, 
#                    label=f'Pico {i+1}' if i==0 else "")
        
#         # C) Análisis Numérico (Spline Local y Derivada)
#         sp_local = my_splines.spline_cubico_natural(xb_local, yb_local)
#         punto_eval = xb_local[2]
        
#         # Derivada de referencia (usando h muy pequeño)
#         deriv_ref = my_diff.central_der_5steps(sp_local, 1e-9, punto_eval)
        
#         # D) Imprimir Tabla de Convergencia
#         print("\n" + "-"*90)
#         print(f"  ANÁLISIS DE PICO #{i+1}: {fecha_evento} (Índice {idx_centro})")
#         print("-" * 90)
#         print(f"  * Bloque: {yb_local}")
#         print(f"  * Tendencia Real (f'): {deriv_ref:.4f}")
#         print("-" * 90)
#         print(f"  {'h (Paso)':<12} | {'Derivada Calc':<15} | {'E. Abs (en)':<15} | {'E. Norm (C)':<18} | {'E. Rel (er)':<15}")
#         print("-" * 90)
        
#         h_real = 1.0/N_TOTAL
#         # Pasos de prueba: grandes, el real, y pequeños (donde explota)
#         pasos_prueba = [0.1, 0.05, 0.02, h_real, 0.005]
        
#         for h in pasos_prueba:
#             # Calcular derivada aproximada
#             d_calc = my_diff.central_der_5steps(sp_local, h, punto_eval)
            
#             # Calcular errores con tu módulo
#             e_n, e_norm, e_rel = my_diff.calcular_errores(deriv_ref, d_calc, h, orden_metodo=4)
            
#             print(f"  {h:<12.6f} | {d_calc:<15.6f} | {e_n:<15.8f} | {e_norm:<18.6f} | {e_rel:<15.6f}")

#     # 4. Guardar la Gráfica de Picos
#     plt.title(f"Puntos de Cambio Drástico (Picos) - {nombre_archivo}")
#     plt.xlabel("Tiempo Normalizado")
#     plt.ylabel("Víctimas")
#     plt.legend()
#     plt.savefig(f"Grafica_Picos_{nombre_archivo}.pdf")
#     plt.close()
    
#     print(f"  -> Gráfica guardada: Grafica_Picos_{nombre_archivo}.pdf")
#     print("="*90)

# print("\n¡ANÁLISIS COMPLETADO EXITOSAMENTE!")


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
    Encuentra los índices de los 3 cambios más drásticos (mayor pendiente).
    Ignora los primeros 2 y últimos 2 puntos para permitir bloques de 5.
    Retorna: Lista de 3 índices ordenados cronológicamente.
    """
    diferencias = np.abs(np.diff(y_data))
    candidatos = []
    
    # Rango seguro: desde índice 2 hasta N-3 para tener vecinos suficientes
    for i in range(2, len(y_data) - 3):
        mag = diferencias[i] 
        candidatos.append((i, mag))
    
    # Ordenar por magnitud del cambio (de mayor a menor)
    candidatos.sort(key=lambda x: x[1], reverse=True)
    
    # Devolver los índices de los top 3
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

    # 1. Lagrange Bloques (Completo)
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

    # 2. Splines Cúbicos (Completo)
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

    # 1. Runge (Original)
    guardar("1_Runge", "1. Fenómeno de Runge", lambda: (
        plt.scatter(x_demo, y_demo, c='k', label='Datos'),
        plt.plot(x_lin_runge, y_lin_runge, c='r', label='Lagrange Global'),
        plt.ylim(min(y_demo)*0.8, max(y_demo)*1.2)
    ))

    # 2. Lagrange Bloques (Original)
    guardar("2_Lagrange_Bloques", "2. Lagrange Bloques", lambda: (
        plt.scatter(x_full, y_full, c='k', s=10, label='Todos los Datos'),
        plt.plot(x_lin_blocks_full, y_lin_blocks_full, c='g', linewidth=1, label='Lagrange Bloques')
    ))

    # 3. COMPARACIÓN (NUEVA)
    guardar("3_Lagrange_comparacion", "3. Comp. Lagrange (Global vs Bloques)", lambda: (
        plt.scatter(x_demo, y_demo, c='k', alpha=0.3, label='Datos (Tramo Global)'),
        plt.plot(x_lin_runge, y_lin_runge, c='r', linestyle='--', alpha=0.6, label='Global (Runge)'),
        plt.plot(x_lin_blocks_full, y_lin_blocks_full, c='g', linewidth=1.5, label='Por Bloques'),
        # Usamos límites basados en los datos reales para que la gráfica global no "aplaste" a la de bloques
        plt.ylim(min(y_full)*0.8, max(y_full)*1.2)
    ))

    # 4. Splines (Antes era 3)
    guardar("4_Splines", "4. Splines Cúbicos", lambda: (
        plt.scatter(x_full, y_full, c='k', s=10, label='Todos los Datos'),
        plt.plot(x_lin_spline_full, y_lin_spline_full, c='b', linewidth=1.5, label='Splines Cúbicos')
    ))

    # 5. Predicción (Antes era 4)
    guardar("5_Prediccion", "5. Estimación Futura", lambda: (
        plt.scatter(x_train[-20:], y_train[-20:], c='k', label='Historia Reciente'),
        plt.scatter(x_test, y_test, c='g', marker='*', s=180, label='Realidad', zorder=10),
        plt.plot(x_test, y_pred_spline, c='purple', linestyle='--', linewidth=2, label='Splines'),
        plt.plot(x_test, y_pred_lagrange, c='orange', linestyle='-.', linewidth=2, label='Lagrange')
    ))

    # ===================================================================
    # ANÁLISIS DE PICOS (CAMBIOS DRÁSTICOS)
    # ===================================================================
    print("  -> Generando análisis de picos drásticos...")

    # 1. Encontrar los picos
    indices_criticos = encontrar_top_3_cambios(y_raw)
    print(f"     Picos detectados en índices: {indices_criticos}")

    # 2. Configurar la Gráfica de Picos
    plt.figure(figsize=(12, 6))
    plt.scatter(x_full, y_full, c='k', s=15, label='Datos Reales')
    plt.plot(x_lin_spline_full, y_lin_spline_full, c='b', alpha=0.5, label='Splines')

    # 3. Iterar sobre los 3 picos para analizar y marcar
    for i, idx_centro in enumerate(indices_criticos):
        
        # A) Definir bloque local
        idx_inicio = idx_centro - 2
        idx_fin = idx_centro + 3 
        xb_local = x_norm[idx_inicio : idx_fin]
        yb_local = y_raw[idx_inicio : idx_fin]
        
        fecha_evento = df['Fecha'].iloc[idx_centro].strftime('%Y-%B')
        
        # B) Marcar el pico en la gráfica con una X roja grande
        plt.scatter(xb_local[2], yb_local[2], c='red', s=150, marker='X', zorder=10, 
                    label=f'Pico {i+1}' if i==0 else "")
        
        # C) Análisis Numérico (Spline Local y Derivada)
        sp_local = my_splines.spline_cubico_natural(xb_local, yb_local)
        punto_eval = xb_local[2]
        
        # Derivada de referencia (usando h muy pequeño)
        deriv_ref = my_diff.central_der_5steps(sp_local, 1e-9, punto_eval)
        
        # D) Imprimir Tabla de Convergencia
        print("\n" + "-"*90)
        print(f"  ANÁLISIS DE PICO #{i+1}: {fecha_evento} (Índice {idx_centro})")
        print("-" * 90)
        print(f"  * Bloque: {yb_local}")
        print(f"  * Tendencia Real (f'): {deriv_ref:.4f}")
        print("-" * 90)
        print(f"  {'h (Paso)':<12} | {'Derivada Calc':<15} | {'E. Abs (en)':<15} | {'E. Norm (C)':<18} | {'E. Rel (er)':<15}")
        print("-" * 90)
        
        h_real = 1.0/N_TOTAL
        # Pasos de prueba: grandes, el real, y pequeños (donde explota)
        pasos_prueba = [0.1, 0.05, 0.02, h_real, 0.005]
        
        for h in pasos_prueba:
            # Calcular derivada aproximada
            d_calc = my_diff.central_der_5steps(sp_local, h, punto_eval)
            
            # Calcular errores con tu módulo
            e_n, e_norm, e_rel = my_diff.calcular_errores(deriv_ref, d_calc, h, orden_metodo=4)
            
            print(f"  {h:<12.6f} | {d_calc:<15.6f} | {e_n:<15.8f} | {e_norm:<18.6f} | {e_rel:<15.6f}")

    # 4. Guardar la Gráfica de Picos (RENOMBRADA Y RENUMERADA)
    plt.title(f"6. Puntos de Cambio Drástico (Picos) - {nombre_archivo}")
    plt.xlabel("Tiempo Normalizado")
    plt.ylabel("Víctimas")
    plt.legend()
    # Se ha renombrado como solicitaste (ajustando el número a 6 por la secuencia: 1,2,3,4,5...)
    plt.savefig(f"Grafica_6_Puntos_criticos_{nombre_archivo}.pdf")
    plt.close()
    
    print(f"  -> Gráfica guardada: Grafica_6_Puntos_criticos_{nombre_archivo}.pdf")
    print("="*90)

print("\n¡ANÁLISIS COMPLETADO EXITOSAMENTE!")