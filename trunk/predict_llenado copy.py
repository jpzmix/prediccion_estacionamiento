import numpy as np
from sklearn.linear_model import LinearRegression
import training
import dataset

X, y = dataset.generar_datos(capacidad_maxima, hora_inicio_llenado, duracion_llenado)
#model = training.entrenar_modelo(X,y)

def entrenar_modelo(X, y):
    """
    Esta función entrena un modelo de regresión lineal con los datos de entrada X y las etiquetas y.
    
    Parámetros:
    - X: matriz de características (número de muestras, número de características).
    - y: vector de etiquetas (número de muestras,).
    
    Retorna:
    - model: el modelo de regresión lineal entrenado.
    """
    # Crear una instancia del modelo de regresión lineal
    model = LinearRegression()

    # Entrenar el modelo con los datos de entrada y las etiquetas
    model.fit(X, y)
    
    return model



def predecir_llenado_estacionamiento(modelo, capacidad_estacionamiento, horas):
    """
    Esta función utiliza un modelo entrenado para predecir cuándo el estacionamiento estará lleno.
    
    Parámetros:
    - modelo: el modelo de regresión lineal entrenado.
    - capacidad_estacionamiento: capacidad máxima del estacionamiento en número de automóviles.
    - horas: horas del día para las cuales hacer la predicción.
    
    Retorna:
    - hora_predicha: la hora estimada en la que el estacionamiento estará lleno.
    """
    # Generar características para las horas del día
    X_prediccion = horas.reshape(-1, 1)

    # Calcular la capacidad restante del estacionamiento para cada hora del día
    capacidad_restante = capacidad_estacionamiento - modelo.predict(X_prediccion)
    
    # Encontrar la primera hora en la que la capacidad restante sea menor o igual a cero
    hora_predicha = np.argmax(capacidad_restante <= 0)
    
    return hora_predicha

# Ejemplo de uso:
# Supongamos que tenemos X e y obtenidos previamente y una capacidad de estacionamiento de 9000 automóviles
modelo = entrenar_modelo(X, y)
capacidad_estacionamiento = 9000
horas = np.arange(24)
hora_predicha = predecir_llenado_estacionamiento(modelo, capacidad_estacionamiento, horas)
print("El estacionamiento estará lleno aproximadamente a las", hora_predicha, "horas.")


