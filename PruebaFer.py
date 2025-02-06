from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Carga el modelo entrenado
model = YOLO("runs/detect/train3/weights/best.pt")#train3v8 y train4v11

# Abre la cámara web
results = model(source=1, show=True)  # Cambia source=0 para usar la cámara web
