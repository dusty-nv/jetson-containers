#!/usr/bin/env python3
import os
import time
import PIL
import clip
import torch

import torchvision
import torchvision.transforms as transforms

import numpy as np

from .utils import AttrDict, load_image, download_model, print_table


class CLIPEmbedding():
    """
    CLIP feature extractor and projector for generating image embeddings.
    """
    def __init__(self, model='ViT-L/14@336px', dtype=np.float32, crop=True, model_cache='/data/models/clip', **kwargs):
        """
        Parameters:
        
          model (str) -- name or path to CLIP model, one of:
                           'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
                           'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        """                
        self.config = AttrDict(name=model)
        self.image_stats = AttrDict()
        self.text_stats = AttrDict()
        self.extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stream = None
        
        dtype = np.dtype(dtype)
        
        if dtype == np.float32:
            self.config.dtype = torch.float32
        elif dtype == np.float16:
            self.config.dtype = torch.float16
        else:
            raise ValueError(f"unsupported datatype:  {dtype}")
            
        print(f'-- loading CLIP {model}')
        
        self.model, _ = clip.load(
            model, 
            device=self.device, 
            jit=False, 
            download_root=model_cache
        )

        self.config.crop = crop
        self.config.input_shape = (self.model.visual.input_resolution, self.model.visual.input_resolution)
        
        self.image_model = self.model.visual

        """
        # TensorRT disabled right now for 8.4 - needs PyTorch 2.1, onnxruntime, not much faster, wrong FP16 results
        trt_path = os.path.join(model_cache, model.replace('/','-').replace('@','-') + '-trt.pth')

        if os.path.isfile(trt_path):
            print(f"-- loading TensorRT model from {trt_path}")
            self.image_model = torch2trt.TRTModule()
            self.image_model.load_state_dict(torch.load(trt_path))
        else:
            # needs PyTorch 2.1 and onnxruntime
            self.image_model = torch2trt.torch2trt(
                self.model.visual.cpu().float(),  # put on CPU for onnx export
                [torch.ones(1, 3, *self.config.input_shape, dtype=torch.float32)],  # TRT expects FP32 input
                fp16_mode=(self.config.dtype == torch.float16),
                log_level=tensorrt.Logger.VERBOSE,
                use_onnx=True,
                onnx_opset=14,
            )
            print(f"-- saving TensorRT model to {trt_path}")
            torch.save(self.image_model.state_dict(), trt_path)
        """
        
        # Pre-processing is able to use GPU with torchvision (cropping is optional)
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L79
        self.preprocessor = torch.nn.Sequential()

        self.preprocessor.append(
            transforms.Resize(
                self.config.input_shape[0] if crop else self.config.input_shape, 
                interpolation=transforms.InterpolationMode.BICUBIC# BILINEAR
            )
        )
        
        if crop:
            self.preprocessor.append(transforms.CenterCrop(self.config.input_shape[0]))
            print("-- image cropping enabled")
            
        self.preprocessor.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.preprocessor.append(transforms.ConvertImageDtype(self.config.dtype))
        
        self.preprocessor = self.preprocessor.eval().to(self.device)

        print(self.model)
        
        print(f"-- {self.config.name} warmup")
        for i in range(2):
            self.embed_image(PIL.Image.new('RGB', self.config.input_shape, (255,255,255)))
        print_table(self.config)
        
    def embed_image(self, image, return_tensors='pt', **kwargs):
        if isinstance(image, str):
            image = load_image(image) #api='torchvision')  # torchvision not any faster, and needs built with PNG

        time_begin_pre = time.perf_counter()

        with torch.cuda.StreamContext(self.stream), torch.inference_mode():
            if isinstance(image, PIL.Image.Image) or isinstance(image, np.ndarray):
                image = transforms.functional.to_tensor(image)
            elif isinstance(image, torch.Tensor):
                pass
            elif hasattr(image, '__cuda_array_interface__'):
                image = torch.as_tensor(image, device='cuda')
            else:
                raise TypeError(f"image was not PIL.Image, np.ndarray, torch.Tensor, or __cuda_array_interface__ (was {type(image)}")
            #else:
            #    image = image.to(device=self.device, dtype=self.config.dtype) / 255.0  # needed when load_image(api='torchvision')
            
            image = image.to(device=self.device, dtype=self.config.dtype)            
            image = self.preprocessor(image).unsqueeze(0)

            time_begin_enc = time.perf_counter()
            output = self.image_model(image)  #self.model.encode_image(image)
            output = self.model.logit_scale.exp() * output
            time_end_enc = time.perf_counter()
            self.config.output_shape = output.shape

        self.image_stats.clip_time = time_end_enc - time_begin_pre
        self.image_stats.clip_rate = 1.0 / self.image_stats.clip_time
        self.image_stats.preprocess_time = time_begin_enc - time_begin_pre
        self.image_stats.encode_time = time_end_enc - time_begin_enc
        self.image_stats.input_shape = f"({image.shape[-1]},{image.shape[-2]}) -> {self.config.input_shape}"
        self.image_stats.output_shape = self.config.output_shape

        if return_tensors == 'np':
            return output.detach().cpu().numpy()
        elif return_tensors == 'pt':
            return output
        else:
            raise ValueError(f"return_tensors should be 'np' or 'pt' (was '{return_tensors}')")
            
    def embed_text(self, text, return_tensors='pt', **kwargs):
        if isinstance(text, str) or (isinstance(text, list) and isinstance(text[0], str)):
            time_begin = time.perf_counter()
            text = clip.tokenize(text).to(self.device)
            self.text_stats.tokens = text.shape 
            self.text_stats.tokens_time = time.perf_counter() - time_begin

        time_begin = time.perf_counter()
        
        with torch.cuda.StreamContext(self.stream), torch.inference_mode():
            output = self.model.encode_text(text)
        
        self.text_stats.encode_time = time.perf_counter() - time_begin
        self.text_stats.output_shape = output.shape

        if return_tensors == 'np':
            return output.detach().cpu().numpy()
        elif return_tensors == 'pt':
            return output
        else:
            raise ValueError(f"return_tensors should be 'np' or 'pt' (was '{return_tensors}')")
            