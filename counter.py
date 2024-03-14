import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from collections import defaultdict
import sqlite3
from datetime import datetime
from db_to_csv import generate_csv
# Función para conectar a la base de datos
def connect_to_database(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    return conn, cursor

# Función para crear la tabla si no existe
def create_table(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS AutoEvents (
                        ID INTEGER PRIMARY KEY AUTOINCREMENT,
                        Date TEXT,
                        Time TEXT,
                        Event TEXT
                    )''')

# Función para insertar un registro en la tabla
def insert_event(cursor, date, time, event):
    cursor.execute("INSERT INTO AutoEvents (Date, Time, Event) VALUES (?, ?, ?)", (date, time, event))

# Función para manejar la interrupción y registrar el evento
def interrupt_handler(event):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")


    conn, cursor = connect_to_database("auto_events.db")


    create_table(cursor)


    insert_event(cursor, current_date, current_time, event)


    conn.commit()
    conn.close()

# Inicializar la captura de video desde la cámara.
cap = cv2.VideoCapture(0)
model = YOLO("best.pt")
# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara. Asegúrate de que esté correctamente conectada.")
    exit()

names = model.model.names
print(f"Clases: {names}")
# Define region points
region_points = [(20, 1000), (1080, 404), (1080, 360), (20, 360)]

# Configurar el contador de objetos
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

class_counts = defaultdict(int)

# Variables para almacenar los valores anteriores de in_count y out_count
prev_in_count = 0
prev_out_count = 0

# Bucle principal para capturar y mostrar el video en tiempo real
while cap.isOpened():
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()
    # Verificar si se capturó correctamente el fotograma
    if not ret:
        print("No se pudo recibir el fotograma. Saliendo...")
        break

    # Realizar el seguimiento de los objetos en el fotograma
    tracks = model.track(frame, persist=True, show=False)

    # Contar y dibujar los objetos detectados en el fotogram
    frame = counter.start_counting(frame, tracks)
    in_count = counter.in_counts
    out_count = counter.out_counts
    print(f'IN: {in_count}  OUT: {out_count}')

    # Comprobar si ha cambiado el valor de in_count 
    if in_count != prev_in_count:
        interrupt_handler("in")
        prev_in_count = in_count

    # Comprobar si ha cambiado el valor de out_count 
    if out_count != prev_out_count:
        interrupt_handler("out")
        prev_out_count = out_count

    # Comprobar si se presiona la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

# Llamar a la función para generar el archivo CSV
generate_csv("auto_events.db", "auto_events.csv")

