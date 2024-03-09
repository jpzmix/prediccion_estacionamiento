import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def entrenar_modelo(X, y):
    """
    Esta función entrena un modelo de regresión polinómico con los datos de entrada X y las etiquetas y.
    
    Parámetros:
    - X: matriz de características (número de muestras, número de características).
    - y: vector de etiquetas (número de muestras,).
    
    Retorna:
    - model: el modelo de regresión polinómico entrenado.
    """
    # Crear un modelo de regresión polinómico de grado 3
    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

    # Entrenar el modelo con los datos de entrada y las etiquetas
    model.fit(X, y)
    
    return model




