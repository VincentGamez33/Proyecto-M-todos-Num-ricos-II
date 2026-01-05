import numpy as np

# ========================================================
# Definición del Spline Cúbico Natural
# ========================================================
def spline_cubico_natural(xs, ys): # xs = [x0,x1,...,xn], ys = [y0,y1,...,yn]
    # Calcula la cantidad de intervalos (n) basado en la longitud de xs
    n = len(xs) - 1
    
    # Calcula las diferencias h_i = x_{i+1} - x_i para cada intervalo
    h = np.diff(xs) 
    
    # Inicializamos el vector alpha para el sistema de ecuaciones
    alpha = np.zeros(n)

    # ----------------------------
    # Ciclo para calcular alpha[i] desde i \in {1,2,...,n-1}
    for i in range(1, n):
        # Calcula el primer término: (3/h_i) * (a_{i+1} - a_i)
        term1 = (3.0 / h[i]) * (ys[i+1] - ys[i])
        # Calcula el segundo término: (3/h_{i-1}) * (a_i - a_{i-1})
        term2 = (3.0 / h[i-1]) * (ys[i] - ys[i-1])
        
        # Asigna el valor a alpha[i]
        alpha[i] = term1 - term2
    # Termina el ciclo
    # ----------------------------

    # Inicializamos listas auxiliares l, mu, z para resolver el sistema tridiagonal
    # (Algoritmo de Crout / Thomas)
    l = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    # Paso 2: Condiciones de frontera natural (S''(x0) = 0)
    l[0] = 1.0
    mu[0] = 0.0
    z[0] = 0.0

    # ----------------------------
    # Ciclo para resolver el sistema lineal tridiagonal i \in {1,2,...,n-1}
    for i in range(1, n):
        # Calcula l_i
        l[i] = 2 * (xs[i+1] - xs[i-1]) - (h[i-1] * mu[i-1])
        # Calcula mu_i
        mu[i] = h[i] / l[i]
        # Calcula z_i
        z[i] = (alpha[i] - (h[i-1] * z[i-1])) / l[i]
    # Termina el ciclo
    # ----------------------------

    # Paso 3: Condición de frontera natural en el extremo derecho (S''(xn) = 0)
    l[n] = 1.0
    z[n] = 0.0
    
    # Inicializamos los coeficientes del spline: a, b, c, d
    # S_j(x) = a_j + b_j(x - x_j) + c_j(x - x_j)^2 + d_j(x - x_j)^3
    a = np.array(ys)      # a_j = y_j
    b = np.zeros(n)
    c = np.zeros(n + 1)
    d = np.zeros(n)

    c[n] = 0.0 # c_n = 0 (condición natural)

    # ----------------------------
    # Ciclo de sustitución hacia atrás j \in {n-1, n-2, ..., 0}
    for j in range(n - 1, -1, -1):
        # Calcula c_j = z_j - mu_j * c_{j+1}
        c[j] = z[j] - (mu[j] * c[j+1])
        
        # Calcula b_j = (a_{j+1} - a_j)/h_j - h_j*(c_{j+1} + 2c_j)/3
        b[j] = (a[j+1] - a[j]) / h[j] - (h[j] * (c[j+1] + 2 * c[j])) / 3.0
        
        # Calcula d_j = (c_{j+1} - c_j) / (3h_j)
        d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
    # Termina el ciclo
    # ----------------------------

    # ========================================================
    # Definición de la función evaluadora interna
    # ========================================================
    def evaluador(z): # z es el punto o lista de puntos a evaluar
        # Convertimos z a un array de numpy por si es un escalar
        z = np.atleast_1d(z)
        resultados = np.zeros_like(z, dtype=float)
        
        # Iteramos sobre cada valor a evaluar
        for k, val in enumerate(z):
            # Encontrar el índice j tal que x_j <= val < x_{j+1}
            if val < xs[0]:
                j = 0
            elif val >= xs[-1]:
                j = n - 1
            else:
                for idx in range(n):
                    if xs[idx] <= val < xs[idx+1]:
                        j = idx
                        break
                else:
                    j = n - 1 

            dx = val - xs[j]
            resultados[k] = a[j] + (b[j] * dx) + (c[j] * dx**2) + (d[j] * dx**3)
            
        # Devuelve el resultado
        if resultados.size == 1:
            return resultados.item(0)
        return resultados

    return evaluador