import numpy as np
from sklearn.linear_model import LinearRegression
import training
import dataset

X, y = dataset.generar_datos()
model = training.entrenar_modelo(X,y)



def estimar_carros(modelo, hora):
    """
    Esta función toma un modelo de regresión lineal entrenado y una hora específica,
    y estima el número de carros que habrá en el estacionamiento a esa hora.
    
    Parámetros:
    - modelo: el modelo de regresión lineal entrenado.
    - hora: la hora específica para la que se quiere estimar el número de carros.
    
    Retorna:
    - carros_estimados: el número estimado de carros en el estacionamiento a la hora dada.
    """
    # Convertir la hora proporcionada por el usuario a un formato adecuado para hacer la predicción
    hora_input = np.array([[hora, hora]])  # Utilizamos la misma hora para el número de carros que entran y salen

    # Hacer la predicción utilizando el modelo entrenado
    carros_estimados = modelo.predict(hora_input)
    
    return carros_estimados

# Ejemplo de uso:
# Supongamos que tenemos el modelo entrenado almacenado en una variable llamada modelo_entrenado
hora_usuario = int(input("Introduce la hora para la que deseas estimar el número de carros (0-23): "))
numero_carros_estimado = estimar_carros(model, hora_usuario)
print(f"Se estima que habrá aproximadamente {numero_carros_estimado[0]:.2f} carros en el estacionamiento a la hora {hora_usuario}.")
