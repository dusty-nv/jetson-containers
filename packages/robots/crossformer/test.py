#!/usr/bin/env python3
from crossformer.model.crossformer_model import CrossFormerModel
model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer")
print(model.get_pretty_spec())