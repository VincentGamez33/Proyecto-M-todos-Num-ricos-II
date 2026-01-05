import numpy as np
import pandas as pd
import math

# ========================================================
# Definición de la aproximación de Lagrange
# ========================================================
def l_nj(xs,j,z): # xs = [x0,x1,...,xn]
    # Calcula la longitud de la lista de puntos xs
    n = len(xs)
    # Inicializamos el numerador (p1) y el denominador (p2) del producto
    p1 = 1.0
    p2 = 1.0

    # ----------------------------
    # Ciclo para el producto desde k \in {0,1,2,...,n}
    for k in range(n):
        # La condición k \neq j es fundamental en la definición de L_{n,j}
        if k != j:
            p1 = p1 * (z - xs[k])     # p1 = (z-x0)*(z-x1)*...*(z-xj-1)*(z-xj+1)*...*(z-xn)
            p2 = p2 * (xs[j] - xs[k]) # p2 = (xj-x0)*(xj-x1)*...*(xj-xj-1)*(xj-xj+1)*...*(xj-xn)
    # Termina el ciclo
    # ----------------------------
            
    # Devuelve el cociente, que es L_{n,j}(z)
    return p1 / p2

# ========================================================
# Definición de los Polinomios de Lagrange
# ========================================================
def lagrange_polynomials(xs,ys,z): # xs = [x0,x1,...,xn], ys = [y0,y1,...,yn]
    # Inicializando las variables para poder entrar al ciclo 'for'.
    n = len(xs)      # Calcula la longitud de la lista de puntos xs.
    ls = np.zeros(n) # 'ls' almacenará los valores de L_{n,j}(z) para cada j.
    p = 0.0          # 'p' es el acumulador para la suma. p = P(z).

    # ----------------------------
    # Ciclo para la sumatoria desde j \in {0,1,2,...,n}.
    for j in range(n):
        # Calcula el j-ésimo polinomio base evaluado en z.
        ls[j] = l_nj(xs, j, z) # L_{n,j}(z).

        # Multiplica por su 'y' correspondiente (y_j) y lo suma al total.
        p = (ys[j] * ls[j]) + p
    # Termina el ciclo
    # ----------------------------
        
    # Devuelve el valor final de la sumatoria P(z).
    return p
