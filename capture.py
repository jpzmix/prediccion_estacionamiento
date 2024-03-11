import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from collections import defaultdict


# Inicializar la captura de video desde la cámara de la laptop
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
    print (f'IN: {in_count}  OUT: {out_count}')

    # Comprobar si se presiona la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
in_count = counter.in_counts
out_count = counter.out_counts
print (f'IN: {in_count}  OUT: {out_count}')
