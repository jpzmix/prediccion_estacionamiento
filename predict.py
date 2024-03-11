import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Cargar los datos desde el archivo CSV
df = pd.read_csv("auto_events_3days.csv")

# Convertir la columna 'Time' a tipo de datos datetime y extraer la hora
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

# Filtrar las entradas y salidas
entradas = df[df['Event'] == 'in']
salidas = df[df['Event'] == 'out']

# Calcular el promedio de entradas y salidas por hora
promedio_entradas_por_hora = entradas.groupby('Time').size() / len(df['Date'].unique())
promedio_salidas_por_hora = salidas.groupby('Time').size() / len(df['Date'].unique())

# Crear un DataFrame para almacenar los resultados
hourly_counts = pd.DataFrame()

# Calcular el número de espacios disponibles por hora (considerando una capacidad máxima de 10 autos)
for hour in range(24):
    if hour == 0:
        hourly_counts.loc[hour, 'available_spaces'] = 10
    else:
        # Calcular los espacios disponibles como la suma de los espacios disponibles en la hora anterior
        # más los promedios de entradas y menos los promedios de salidas para la hora actual
        available_spaces = (hourly_counts.loc[hour - 1, 'available_spaces'] -
                            promedio_entradas_por_hora.get(hour, 0) +
                            promedio_salidas_por_hora.get(hour, 0))
        
        # Asegurar que los espacios disponibles estén dentro del rango [0, capacidad_max]
        hourly_counts.loc[hour, 'available_spaces'] = min(max(available_spaces, 0), 10)

# Suponiendo que hourly_counts es tu DataFrame con una sola columna
hourly_counts_vector = hourly_counts['available_spaces'].to_numpy()


# Crear el modelo de bosque aleatorio
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
X = np.arange(24).reshape(-1, 1)
# Entrenar el modelo
modelo.fit(X, hourly_counts_vector)

# Hacer predicciones
predicciones = modelo.predict(X)

# Redondear las predicciones hacia abajo y convertirlas a enteros
predicciones_redondeadas = np.floor(predicciones).astype(int)

# Crear un DataFrame con las columnas hora, available_spaces y predicciones
resultados = pd.DataFrame({
    'hora': np.arange(24),
    'espacios disponibles(real)': hourly_counts['available_spaces'].astype(int),
    'espacios disponibles(predict)': predicciones_redondeadas
})

# Imprimir el DataFrame
print(resultados)

# Guardar los resultados en un archivo CSV
resultados.to_csv("resultados_prediccion.csv", index=False)


