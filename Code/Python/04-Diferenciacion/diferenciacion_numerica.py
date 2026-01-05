import numpy as np
import pandas as pd
import math

# ========================================================
# Definición de la aproximación para la Derivada hacia atrás
# ========================================================
def backward_der_5steps(f,h,b):
    numerador = 25*f(b) - 48*f(b-h) + 36*f(b-(2*h)) - 16*f(b-(3*h)) + 3*f(b-(4*h))
    denominador = 12 * h
    return numerador/denominador

# ========================================================
# Definición de la aproximación para la Derivada centrada
# ========================================================
def central_der_5steps(f,h,x):
    numerador = f(x-(2*h)) - 8*f(x-h) + 8*f(x+h) - f(x+(2*h))
    denominador = 12 * h
    return numerador/denominador

# ========================================================
# Definición de la aproximación para la Derivada hacia adelante
# ========================================================
def forward_der_5steps(f,h,a):
    numerador = -25*f(a) + 48*f(a+h) - 36*f(a+(2*h)) + 16*f(a+(3*h)) - 3*f(a+(4*h))
    denominador = 12 * h
    return numerador/denominador

# ========================================================
# Definición de los errores
# ========================================================
def calcular_errores(valor_real, valor_aprox, h, orden_metodo=4):
    # Error Absoluto (en)
    e_n = abs(valor_real - valor_aprox)
    
    # Error Normalizado (e_N,h = en / h^k)
    # Buscamos que este valor tienda a una constante C
    if h > 0:
        e_norm = e_n / (h ** orden_metodo)
    else:
        e_norm = 0.0
        
    # Error Relativo (er,n = en / |aprox|)
    # Nota: La imagen indica dividir por el valor APROXIMADO (app)
    if abs(valor_aprox) > 1e-12:
        e_rel = e_n / abs(valor_aprox)
    else:
        e_rel = 0.0
        
    return e_n, e_norm, e_rel
