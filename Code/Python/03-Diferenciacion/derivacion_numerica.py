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
