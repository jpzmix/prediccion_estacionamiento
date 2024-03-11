import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Definir los datos de entrada (horas del día) y de salida (disponibilidad de espacios)
X = np.arange(24).reshape(-1, 1)
y = np.array([
    10., 10., 10., 10., 10., 7., 4., 0., 3., 3., 6., 3.,
    0., 0., 0., 1., 1., 3., 6., 6., 10., 10., 10., 10.
])

# Crear el modelo de bosque aleatorio
modelo = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
modelo.fit(X, y)

# Hacer predicciones
predicciones = modelo.predict(X)

# Imprimir las predicciones
print(predicciones)
