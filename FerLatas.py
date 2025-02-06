from ultralytics import YOLO
import torch

def main():
    # Verifica que PyTorch esté utilizando CUDA
    print("PyTorch CUDA disponible:", torch.cuda.is_available())
    print("Versión de CUDA:", torch.version.cuda)

    # Carga el modelo
    model = YOLO("yolo11n.pt")

    # Entrena el modelo
    results = model.train(
        data="C:\\Users\\Ricardo\\Downloads\\can.v2i.yolov8\\data.yaml",
        epochs=100,
        imgsz=640,
        device=0
    )

if __name__ == "__main__":
    main()
