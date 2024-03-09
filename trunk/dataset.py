import numpy as np

import numpy as np


def generar_datos_promedio():
    """
    Genera datos sintéticos para simular el llenado promedio del estacionamiento a lo largo del día,
    con algunas horas pico opcionales.
    
    Parámetros:
    - capacidad_maxima: capacidad máxima del estacionamiento.
    - horas_pico: lista de horas pico en las que el número de carros es mayor.
    - num_autos_pico: número de carros en horas pico.
    - promedio_diario: número promedio de carros durante el resto del día.
    
    Retorna:
    - X: matriz de características (horas del día).
    - y: vector de etiquetas (número de carros en el estacionamiento para cada hora).
    """
    # Ejemplo de uso:
    capacidad_maxima = 100
    horas_pico = [12,13,14]   # Horas pico
    num_autos_pico = 100  # Número de carros en horas pico
    promedio_diario = 30
    horas = np.arange(24)
    y = np.zeros(24)
    
    # Llenado promedio del estacionamiento durante el resto del día
    y[:] = promedio_diario
    
    # Llenado durante las horas pico
    for hora in horas_pico:
        if hora in horas:
            y[hora] = num_autos_pico
    
    # Asegurarse de que el número de carros no exceda la capacidad máxima
    y[y > capacidad_maxima] = capacidad_maxima
    
    X = horas.reshape(-1, 1)
    
    # Imprimir los valores de X y y
    print("Valores de X:")
    print(X)
    print("\nValores de y:")
    print(y)

    return X, y







