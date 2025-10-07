#---
# name: openai
# group: llm
# depends: [python]
# test: test.py
# notes: https://github.com/openai/openai-python
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN uv pip install openai pycurl
RUN python3 -c 'import openai; print(f"OpenAI API version: {openai.__version__}")'

# https://github.com/ggerganov/llama.cpp/discussions/795
# https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
# https://github.com/mckaywrigley/chatbot-ui
# https://llm.mlc.ai/docs/deploy/rest.html
# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
