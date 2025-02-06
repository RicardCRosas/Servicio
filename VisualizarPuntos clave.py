from ultralytics import YOLO
import cv2
import numpy as np

# Carga el modelo entrenado
model = YOLO("runs/pose/train4/weights/best.pt")

# Carga la imagen de prueba
image_path = r"C:\Users\Ricardo\Downloads\CHIHUAHUA.jpg"
image = cv2.imread(image_path)

# Verifica si la imagen se cargó correctamente
if image is None:
    print(f"Error al cargar la imagen desde la ruta: {image_path}")
    exit()

# Realiza la inferencia usando la imagen
results = model.predict(source=image, save=False, conf=0.5)

# Verifica si hay resultados y si hay puntos clave
if not results or results[0].keypoints is None:
    print("No se detectaron puntos clave.")
    exit()

# Obtener puntos clave del primer objeto detectado
keypoints = results[0].keypoints.xy.cpu().numpy()  # Obtiene coordenadas X, Y

# Mostrar la estructura de los keypoints
print("Estructura de keypoints:", keypoints)
print("Forma de keypoints:", keypoints.shape)

# Dibujar puntos clave en la imagen
for point in keypoints[0]:  # Iterar por los puntos del primer objeto detectado
    x, y = int(point[0]), int(point[1])  # Coordenadas X, Y
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Dibuja un círculo verde

# Mostrar la imagen con los puntos clave
cv2.imshow("Puntos clave detectados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
