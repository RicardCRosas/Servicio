from ultralytics import YOLO
import cv2
import numpy as np
import time

# Carga el modelo entrenado
model = YOLO("runs/pose/train4/weights/best.pt")

# Configuración
camera_source = 1  # Cambia a 0 para usar la cámara por defecto
min_confidence = 0.5  # Confianza mínima para considerar un punto clave relevante

# Variables para calcular la velocidad
time_prev = None
nose_prev_position = None

# Función para calcular ángulos entre tres puntos clave
def calculate_angle(p1, p2, p3):
    try:
        # Asegurarse de que los puntos estén en la CPU y convertir a numpy
        a = np.array(p1[:2])
        b = np.array(p2[:2])
        c = np.array(p3[:2])

        # Vectores
        ba = a - b
        bc = c - b

        # Ángulo entre los vectores
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        print(f"Error al calcular el ángulo: {e}")
        return None

# Función para calcular la velocidad del perro
def calculate_speed(nose_current_position, time_current):
    global nose_prev_position, time_prev

    if nose_prev_position is None or time_prev is None:
        nose_prev_position = nose_current_position
        time_prev = time_current
        return 0.0  # Sin velocidad inicial

    # Calcular distancia recorrida
    distance = np.linalg.norm(np.array(nose_current_position) - np.array(nose_prev_position))
    # Calcular tiempo transcurrido
    time_elapsed = time_current - time_prev

    if time_elapsed == 0:
        return 0.0

    # Calcular velocidad (pixeles por segundo)
    speed = distance / time_elapsed

    # Actualizar las posiciones y tiempo previos
    nose_prev_position = nose_current_position
    time_prev = time_current

    return speed

# Función principal para el análisis dinámico
def analyze_pose(keypoints):
    # Definir índices de puntos relevantes según la referencia de puntos
    FRONT_LEFT_PAW, FRONT_LEFT_KNEE, FRONT_LEFT_ELBOW = 0, 1, 2
    REAR_LEFT_PAW, REAR_LEFT_KNEE, REAR_LEFT_ELBOW = 3, 4, 5
    FRONT_RIGHT_PAW, FRONT_RIGHT_KNEE, FRONT_RIGHT_ELBOW = 6, 7, 8
    REAR_RIGHT_PAW, REAR_RIGHT_KNEE, REAR_RIGHT_ELBOW = 9, 10, 11
    NOSE = 16

    # Verificar si los puntos clave requeridos están presentes
    required_points = [FRONT_LEFT_PAW, FRONT_LEFT_KNEE, FRONT_LEFT_ELBOW,
                       REAR_LEFT_PAW, REAR_LEFT_KNEE, REAR_LEFT_ELBOW,
                       FRONT_RIGHT_PAW, FRONT_RIGHT_KNEE, FRONT_RIGHT_ELBOW,
                       REAR_RIGHT_PAW, REAR_RIGHT_KNEE, REAR_RIGHT_ELBOW,
                       NOSE]
    for idx in required_points:
        if idx >= len(keypoints) or keypoints[idx][2] < min_confidence:
            print(f"Punto clave {idx} no disponible o con baja confianza.")
            return None

    # Extraer las coordenadas (x, y) de los puntos relevantes
    front_left_knee = keypoints[FRONT_LEFT_KNEE]
    front_left_elbow = keypoints[FRONT_LEFT_ELBOW]
    rear_left_knee = keypoints[REAR_LEFT_KNEE]
    rear_left_elbow = keypoints[REAR_LEFT_ELBOW]
    front_right_knee = keypoints[FRONT_RIGHT_KNEE]
    front_right_elbow = keypoints[FRONT_RIGHT_ELBOW]
    rear_right_knee = keypoints[REAR_RIGHT_KNEE]
    rear_right_elbow = keypoints[REAR_RIGHT_ELBOW]
    nose = keypoints[NOSE]

    # Calcular ángulos para el lado izquierdo
    front_leg_angle_left = calculate_angle(front_left_knee, front_left_elbow, rear_left_elbow)
    rear_leg_angle_left = calculate_angle(rear_left_knee, rear_left_elbow, front_left_elbow)
    head_angle_left = calculate_angle(nose, rear_left_elbow, front_left_elbow)

    # Calcular ángulos para el lado derecho
    front_leg_angle_right = calculate_angle(front_right_knee, front_right_elbow, rear_right_elbow)
    rear_leg_angle_right = calculate_angle(rear_right_knee, rear_right_elbow, front_right_elbow)
    head_angle_right = calculate_angle(nose, rear_right_elbow, front_right_elbow)

    # Validar y mostrar ángulos del lado izquierdo
    if front_leg_angle_left is not None:
        print(f"Ángulo de la pata delantera izquierda: {front_leg_angle_left:.2f}°")
    else:
        print("No se pudo calcular el ángulo de la pata delantera izquierda.")

    if rear_leg_angle_left is not None:
        print(f"Ángulo de la pata trasera izquierda: {rear_leg_angle_left:.2f}°")
    else:
        print("No se pudo calcular el ángulo de la pata trasera izquierda.")

    if head_angle_left is not None:
        print(f"Ángulo de la cabeza (lado izquierdo): {head_angle_left:.2f}°")
    else:
        print("No se pudo calcular el ángulo de la cabeza (lado izquierdo).")

    # Validar y mostrar ángulos del lado derecho
    if front_leg_angle_right is not None:
        print(f"Ángulo de la pata delantera derecha: {front_leg_angle_right:.2f}°")
    else:
        print("No se pudo calcular el ángulo de la pata delantera derecha.")

    if rear_leg_angle_right is not None:
        print(f"Ángulo de la pata trasera derecha: {rear_leg_angle_right:.2f}°")
    else:
        print("No se pudo calcular el ángulo de la pata trasera derecha.")

    if head_angle_right is not None:
        print(f"Ángulo de la cabeza (lado derecho): {head_angle_right:.2f}°")
    else:
        print("No se pudo calcular el ángulo de la cabeza (lado derecho).")

    # Calcular velocidad de la nariz
    time_current = time.time()
    speed = calculate_speed(nose[:2], time_current)

    print(f"Velocidad del perro: {speed:.2f} píxeles/segundo")

# Procesamiento de la cámara en tiempo real
cap = cv2.VideoCapture(camera_source)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

print("Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Realizar la inferencia de pose
    results = model.predict(source=frame, save=False, conf=min_confidence)

    # Verificar si se detectaron puntos clave
    if len(results) == 0 or results[0].keypoints is None:
        cv2.imshow("Análisis de Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Obtener puntos clave del primer objeto detectado
    keypoints = results[0].keypoints.data[0].cpu().numpy()

    # Analizar la pose
    analyze_pose(keypoints)

    # Dibujar puntos clave en el frame
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > min_confidence:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{i}", (int(x), (int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Mostrar la imagen con el análisis
    cv2.imshow("Análisis de Pose", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
