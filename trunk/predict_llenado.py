import numpy as np
from sklearn.linear_model import LinearRegression
import dataset
import training
import test_data

def entrenar_modelo(X,y):
    """
    Entrena un modelo de regresión lineal con los datos de entrada X y las etiquetas y.
    
    Parámetros:
    - X: matriz de características (número de muestras, número de características).
    - y: vector de etiquetas (número de muestras,).
    
    Retorna:
    - model: el modelo de regresión lineal entrenado.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def encontrar_hora_llenado(modelo, umbral, horas):
    """
    Encuentra la hora en la que el estacionamiento está lleno utilizando el modelo de regresión.
    Imprime el número de carros estimado para cada hora del día.
    
    Parámetros:
    - modelo: el modelo de regresión lineal entrenado.
    - umbral: umbral que indica cuándo el estacionamiento está lleno.
    - horas: vector con las horas del día.
    """
    # Predecir el número de autos en el estacionamiento para cada hora del día
    X_Test,y_test = test_data.generar()
    num_autos = modelo.predict(X_Test)
    
    # Imprimir el número de carros estimado para cada hora del día
    print("Número de carros estimado para cada hora del día:")
    for hora, num_auto in zip(horas, num_autos):
        print(f"Hora: {hora}, Número de carros estimado: {num_auto}")
    
    # Encontrar las horas en las que el número de autos excede el umbral
    horas_llenado = [hora for hora, num_auto in zip(horas, num_autos) if num_auto > umbral]
    
    # Imprimir las horas en las que el estacionamiento está lleno
    if horas_llenado:
        print("\nEl estacionamiento está lleno en las siguientes horas:")
        print(horas_llenado)
    else:
        print("\nEl estacionamiento no está lleno en ninguna hora.")


# Ejemplo de uso:
# Generar datos de entrenamiento (por ejemplo)



horas = np.arange(24)
print (horas)


#capacidad_maxima = 36  # Capacidad máxima del estacionamiento
X,y=dataset.generar_datos_promedio()

# Entrenar el modelo
modelo = entrenar_modelo(X,y)

# Umbral para determinar si el estacionamiento está lleno
umbral = 32
encontrar_hora_llenado(modelo, umbral, horas)