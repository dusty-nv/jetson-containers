#!/usr/bin/env bash
python3 -c 'from nano_llm.plugins import AutoASR, AutoTTS'

python3 -m nano_llm.chat --api mlc \
  --model princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT \
  --quantization q4f16_ft \
  --prompt "What's a good recipe for making tabouli?" \
  --prompt 'How do I allocate memory in C?'
  
python3 -m nano_llm.chat --api=mlc --debug \
  --model Efficient-Large-Model/VILA-2.7b \
  --max-new-tokens 128 \
  --prompt /data/prompts/images.json 
  
python3 -m nano_llm.vision.example --debug
