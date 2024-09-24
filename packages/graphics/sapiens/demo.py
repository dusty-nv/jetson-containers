#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import numpy as np
import colorsys
import matplotlib.colors as mcolors
from jetson_utils import videoSource, videoOutput, cudaFromNumpy, cudaAllocMapped, cudaDeviceSynchronize
from torchvision import transforms
from PIL import Image
from models import SAPIENS_LITE_MODELS_PATH, LABELS_TO_IDS


def get_palette(num_cls):
    palette = [0] * (256 * 3)
    for j in range(1, num_cls):
        hue = (j - 1) / (num_cls - 1)
        saturation = 1.0
        value = 1.0 if j % 2 == 0 else 0.5
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        r, g, b = [int(x * 255) for x in rgb]
        palette[j * 3:j * 3 + 3] = [r, g, b]
    return palette

def create_colormap(palette):
    colormap = np.array(palette).reshape(-1, 3) / 255.0
    return mcolors.ListedColormap(colormap)

def visualize_mask_with_overlay(img_np, mask_np, labels_to_ids, alpha=0.5):
    num_cls = len(labels_to_ids)
    palette = get_palette(num_cls)
    colormap = create_colormap(palette)

    overlay = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for label, idx in labels_to_ids.items():
        if idx != 0:
            overlay[mask_np == idx] = np.array(colormap(idx)[:3]) * 255

    blended = np.uint8(img_np * (1 - alpha) + overlay * alpha)
    return blended

def load_model(task, version):
    model_path = SAPIENS_LITE_MODELS_PATH[task][version]
    if not torch.cuda.is_available():
        print("CUDA no está disponible. Se requiere un dispositivo CUDA para ejecutar este script.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"El archivo del modelo no existe en {model_path}")
        sys.exit(1)
    model = torch.jit.load(model_path)
    model.eval().to("cuda")
    return model

transform_fn = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def segment(frame_np, model):
    pil_image = Image.fromarray(frame_np)
    input_tensor = transform_fn(pil_image).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        preds = model(input_tensor)
    preds = torch.nn.functional.interpolate(preds, size=(frame_np.shape[0], frame_np.shape[1]), mode="bilinear", align_corners=False)
    _, mask = torch.max(preds, 1)
    mask_np = mask.squeeze(0).cpu().numpy()
    return mask_np

if __name__ == "__main__":
    TASK = 'seg'
    VERSION = 'sapiens_0.3b'

    model = load_model(TASK, VERSION)

    parser = argparse.ArgumentParser(description="Real-time segmentation with jetson_utils",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("input", type=str, default="/dev/video0", nargs='?', help="URI del stream de entrada")
    parser.add_argument("output", type=str, default="display://0", nargs='?', help="URI del stream de salida")

    args = parser.parse_args()

    # Crear fuente y salida de video
    input_stream = videoSource(args.input, argv=sys.argv)
    output_stream = videoOutput(args.output, argv=sys.argv)

    while True:
        # Capturar el siguiente frame
        img = input_stream.Capture()

        if img is None:  # timeout
            continue

        img_np = img.numpy()

        mask_np = segment(img_np, model)

        blended_frame = visualize_mask_with_overlay(img_np, mask_np, LABELS_TO_IDS, alpha=0.5)

        output_img = cudaFromNumpy(blended_frame)

        output_stream.Render(output_img)

        output_stream.SetStatus("Segmentación en tiempo real | {:d}x{:d} | {:.1f} FPS".format(
            img.width, img.height, output_stream.GetFrameRate()))

        cudaDeviceSynchronize()

        if not input_stream.IsStreaming() or not output_stream.IsStreaming():
            break
