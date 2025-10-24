#!/usr/bin/env bash

echo "Testing ktransformers version ${KTRANSFORMERS_VERSION}"
wget $WGET_FLAGS https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/main/DeepSeek-V2-Lite.Q4_K_M.gguf -O DeepSeek-V2-Lite-Chat.Q4_K_M.gguf

# Start local chat
python3 -m ktransformers.local_chat --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path ./DeepSeek-V2-Lite-Chat.Q4_K_M.gguf

# If you see “OSError: We couldn't connect to 'https://huggingface.co' to load this file”, try：
# GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
# python  ktransformers.local_chat --model_path ./DeepSeek-V2-Lite --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF

print('KTRANSFORMERS OK\n')
