import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def entrenar_modelo(X, y):
    """
    Esta función entrena un modelo de regresión lineal con los datos de entrada X y las etiquetas y,
    devuelve el modelo entrenado listo para hacer predicciones.
    
    Parámetros:
    - X: matriz de características (número de muestras, número de características).
    - y: vector de etiquetas (número de muestras,).
    
    Retorna:
    - model: el modelo de regresión lineal entrenado.
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear una instancia del modelo de regresión lineal
    model = LinearRegression()

    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)
    
    # Predecir los valores de salida para los datos de prueba
    y_pred = model.predict(X_test)

    # Calcular el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)

    # Calcular el coeficiente de determinación (R^2)
    r2 = r2_score(y_test, y_pred)

    print("Error cuadrático medio:", mse)
    print("Coeficiente de determinación (R^2):", r2)
    
    return model


