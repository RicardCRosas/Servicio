# PyTorch (ejemplo común en implementaciones YOLO)
import torch

# Verificar si hay GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Entrenando en: {device}')
