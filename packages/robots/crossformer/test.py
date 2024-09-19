#!/usr/bin/env python3
from crossformer.model.crossformer_model import CrossFormerModel
import logging

logging.basicConfig(level=logging.DEBUG)
model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")
print(model.get_pretty_spec())