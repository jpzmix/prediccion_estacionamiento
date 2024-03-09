from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

# Inicializar el modelo YOLO
model = YOLO("best.pt")

# Capturar el video de la cámara
cap = cv2.VideoCapture(0)  # 0 para la primera cámara

# Verificar si la cámara se abrió correctamente
assert cap.isOpened(), "Error al abrir la cámara"

# Configurar el contador de objetos
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=None,  # Define reg_pts como None si no necesitas regiones de interés
                 classes_names=model.names,
                 draw_tracks=True)

# Bucle principal para procesar imágenes de la cámara
while cap.isOpened():
    ret, frame = cap.read()  # Leer un fotograma de la cámara

    if not ret:
        print("No se pudo recibir el fotograma. Saliendo...")
        break

    # Realizar el seguimiento de los objetos en el fotograma
    #tracks = model.track(frame, persist=True, show=False)

    # Contar y dibujar los objetos detectados en el fotograma
    #frame = counter.start_counting(frame, tracks)

    # Mostrar el fotograma con los objetos contados
    #cv2.imshow('Object Counting', frame)

    # Comprobar si se presionó la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
