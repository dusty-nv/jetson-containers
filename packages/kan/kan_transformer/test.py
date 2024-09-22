#!/usr/bin/env python3
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