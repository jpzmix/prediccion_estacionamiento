import numpy as np
def generar():
    """
    Genera datos de prueba sintéticos para simular el llenado promedio del estacionamiento a lo largo del día,
    con algunas horas pico opcionales.
    
    Parámetros:
    - capacidad_maxima: capacidad máxima del estacionamiento.
    - horas_pico: lista de horas pico en las que el número de carros es mayor.
    - num_autos_pico: número de carros en horas pico.
    - promedio_diario: número promedio de carros durante el resto del día.
    
    Retorna:
    - X_test: matriz de características de prueba (horas del día).
    - y_test: vector de etiquetas de prueba (número de carros en el estacionamiento para cada hora).
    """
    capacidad_maxima = 50
    horas_pico = [12,13,14]  # Horas pico
    num_autos_pico = 50  # Número de carros en horas pico
    promedio_diario = 30
    horas = np.arange(24)
    y_test = np.zeros(24)
    
    # Llenado promedio del estacionamiento durante el resto del día
    y_test[:] = promedio_diario
    
    # Llenado durante las horas pico
    for hora in horas_pico:
        if hora in horas:
            y_test[hora] = num_autos_pico
    
    # Asegurarse de que el número de carros no exceda la capacidad máxima
    y_test[y_test > capacidad_maxima] = capacidad_maxima
    
    X_test = horas.reshape(-1, 1)
    
    return X_test, y_test