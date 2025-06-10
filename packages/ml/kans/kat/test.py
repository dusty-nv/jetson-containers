#!/usr/bin/env python3
import os
import shutil
import sys

# Ruta de origen y destino
source_file = '/opt/kat/katransformer.py'
destination_dir = '/test/'

# Asegúrate de que el directorio de destino existe
if not os.path.exists(destination_dir):
    print(f"Error: Destination directory {destination_dir} does not exist.")
    sys.exit(1)

# Copiar katransformer.py de /opt/kat/ a /test/
try:
    shutil.copy(source_file, destination_dir)
    print(f"Successfully copied {source_file} to {destination_dir}")
except FileNotFoundError:
    print(f"Error: {source_file} not found.")
    sys.exit(1)

# Cambiar el directorio actual a /test/
os.chdir(destination_dir)
print(f"Current working directory: {os.getcwd()}")

print('testing KAT...')

from urllib.request import urlopen
from PIL import Image
import timm
import torch
import json
import katransformer

# Load the image
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

# Move model to CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained KAT model
model = timm.create_model('hf_hub:adamdad/kat_tiny_patch16_224', pretrained=True)
model = model.to(device)
model = model.eval()

# Get model-specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Preprocess image and make predictions
output = model(transforms(img).unsqueeze(0).to(device))  # unsqueeze single image into batch of 1
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

# Load ImageNet class names
imagenet_classes_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
class_idx = json.load(urlopen(imagenet_classes_url))

# Map class indices to class names
top5_class_names = [class_idx[idx] for idx in top5_class_indices[0].tolist()]

# Print top 5 probabilities and corresponding class names
print("Top-5 Class Names:", top5_class_names)
print("Top-5 Probabilities:", top5_probabilities)

print('KAT OK\n')