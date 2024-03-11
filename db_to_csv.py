import csv
import sqlite3

# Función para generar un archivo CSV con el contenido de la base de datos
def generate_csv(database_name, csv_filename):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # Obtener todos los registros de la tabla
    cursor.execute("SELECT * FROM AutoEvents")
    records = cursor.fetchall()

    # Crear el archivo CSV e escribir los datos
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Escribir el encabezado
        csv_writer.writerow(['ID', 'Date', 'Time', 'Event'])
        
        # Escribir los registros
        csv_writer.writerows(records)

    # Cerrar la conexión a la base de datos
    conn.close()

