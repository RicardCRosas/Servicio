from ultralytics import YOLO
import cv2
import numpy as np

# Carga el modelo entrenado
model = YOLO("runs/pose/train4/weights/best.pt")

# Carga una imagen de prueba
image_path = "C:\\Users\\Ricardo\\Downloads\\CHIHUAHUA.jpg"
image = cv2.imread(image_path)

# Verifica si la imagen se cargó correctamente
if image is None:
    print(f"Error al cargar la imagen desde la ruta: {image_path}")
    exit()

# Realiza la inferencia usando la imagen
results = model.predict(source=image, save=False, conf=0.5)

# Verifica si hay resultados
if len(results) == 0 or results[0].keypoints is None:
    print("No se detectaron puntos clave.")
    exit()

# Obtener puntos clave del primer objeto detectado
keypoints = results[0].keypoints.data[0]  # Usamos .data para acceder a los valores
print(f"Estructura de keypoints: {keypoints}")
print(f"Forma de keypoints: {keypoints.shape}")

# Definir índices relevantes (0 a 22)
relevant_indices = list(range(23))  # Incluye todos los puntos desde 0 hasta 22

# Filtrar puntos clave válidos
min_confidence = 0.5  # Define la confianza mínima
relevant_keypoints = []
for i in relevant_indices:
    if i < keypoints.shape[0]:  # Verifica que el índice esté dentro del rango
        x, y, conf = keypoints[i]
        if conf > min_confidence:  # Filtra por confianza
            relevant_keypoints.append((i, (x, y, conf)))

# Verifica si hay puntos clave relevantes
if not relevant_keypoints:
    print("No se encontraron puntos clave relevantes con confianza suficiente.")
    exit()

print("Puntos clave relevantes:")
for i, (x, y, conf) in relevant_keypoints:
    print(f'Índice: {i}, Coordenadas: ({x}, {y}), Confianza: {conf}')

# Marcar los puntos clave en la imagen
for i, (x, y, conf) in relevant_keypoints:
    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Dibuja un círculo verde en cada punto relevante
    cv2.putText(image, f"{i}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Índice del punto

# Mostrar la imagen con los puntos marcados
cv2.imshow("Puntos clave relevantes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
