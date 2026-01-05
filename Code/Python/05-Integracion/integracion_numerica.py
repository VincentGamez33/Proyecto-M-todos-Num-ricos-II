import numpy as np
import pandas as pd
import math

# ========================================================
# Definición de la aproximación para la Integral de Riemann
# ========================================================
def integral_riemann_uniform(f,a,b,n):
    # Inicializando las variables para poder entrar al ciclo 'for'.
    h = (b-a)/n # Calcula el tamaño del 'paso'.
    SR = 0.0    # Inicializamos la suma en 0.
    xj = a      # Inicia el acumulador 'xj' en el punto 'a'.

    # ----------------------------
    # Ciclo que suma las alturas f(xj) para j \in {1,2,...,n}.
    for j in range(n):
        xj = xj + h  # Avanza al extremo derecho del subintervalo.
        yj =  f(xj)  # Calcula la altura en ese punto.
        SR = SR + yj # Acumula la altura.
    # Termina el ciclo.
    # ----------------------------

    # Devuelve el valor final de la sumatoria S(f,P).
    return h * SR # Calcula el área total al multiplicar por 'h'.

# ========================================================
# Definición de la aproximación para la Regla Trapezoidal
# ========================================================
def integral_trapezoidal_rule_uniform(f, a, b, n):
    # Inicializando las variables para poder entrar al ciclo 'for'.
    h = (b - a) / n          # Calcula el ancho 'h' de cada subintervalo.
    TR = (f(a) + f(b)) / 2.0 # Inicializa la suma con el promedio de los extremos f(a) y f(b).

    # ----------------------------
    # Ciclo para sumar las evaluaciones en los puntos interiores de la sumatoria.
    # Para j \in {1,2,...,n-1}.
    for j in range(1, n):
        xj = a + j * h # Calcula el j-ésimo punto (nodo) interior de la malla
        yj = f(xj)     # Evalúa la función en dicho punto
        TR = TR + yj   # Acumula la evaluación (altura) en la suma total
    # Termina el ciclo
    # ----------------------------

    # Devuelve el valor final de la aproximación.
    return h * TR # Calcula el área total multiplicando la suma acumulada por 'h'.

# ========================================================
# Definición de la aproximación para la Regla de Simpson 1/3
# ========================================================
def integral_simpson_rule_uniform(f, a, b, n):
    # Inicializando las variables para poder entrar al ciclo 'for'.
    h = (b - a) / n # Calcula el ancho 'h' de cada subintervalo.
    SR = 0.0        # Inicializamos la sumatoria en 0.

    # ----------------------------
    # Ciclo que aplica la Regla de Simpson 1/3 simple en cada uno de los 'n' subintervalos.
    # Para j \in {1,2,...,n}
    for j in range(1, n + 1):
        # Calcula los extremos del j-ésimo subintervalo: (x_{j-1}, x_j).
        xjminus1 = a + (j - 1) * h
        xj = a + j * h
        # Calcula el punto medio de dicho subintervalo.
        mj = (xjminus1 + xj) / 2.0
        # Evalúa la fórmula de Simpson simple: f(x_{j-1}) + 4*f(m_j) + f(x_j).
        Aj = f(xjminus1) + 4 * f(mj) + f(xj)
        
        # Acumula la contribución del subintervalo.
        SR = SR + Aj
    # Termina el ciclo.
    # ----------------------------

    # Devuelve el valor final de la aproximación.
    return (h / 3.0) * SR # La fórmula de la regla simple es (h/3) * [...].
