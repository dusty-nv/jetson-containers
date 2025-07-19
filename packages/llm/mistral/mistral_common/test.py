#!/usr/bin/env python3
print('Testing mistral_common...')
from mistral_common.protocol.instruct.messages import ImageURLChunk
from mistral_common.tokens.tokenizers.image import ImageEncoder, ImageConfig, SpecialImageIDs

special_ids = SpecialImageIDs(img=10, img_break=11, img_end=12)  # These are normally automatically set by the tokenizer

config = ImageConfig(image_patch_size=14, max_image_size=224, spatial_merge_size=2)

image = ImageURLChunk(image_url="https://live.staticflickr.com/7250/7534338696_b33e941b7d_b.jpg")

encoder = ImageEncoder(config, special_ids)
encoder(image)

print('mistral_common OK')
