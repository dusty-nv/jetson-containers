#!/usr/bin/env bash
set -ex

git clone https://github.com/oobabooga/text-generation-webui "$OOBABOOGA_ROOT_DIR"
git -C "$OOBABOOGA_ROOT_DIR" checkout "$OOBABOOGA_SHA"

# Fix text-generation-webui requirements
sed -i \
  -e 's|^bitsandbytes.*|#bitsandbytes|g' \
  -e 's|^llama-cpp-python.*|llama-cpp-python|g' \
  -e 's|^exllamav2.*|exllamav2|g' \
  -e 's|^autoawq.*||g' \
  -e 's|^numpy.*|numpy|g' \
  -e 's|^aqlm.*|aqlm|g' \
  -e 's|^transformers.*|transformers|g' \
  -e 's|^https://github.com/turboderp/exllama.*||g' \
  -e 's|^https://github.com/jllllll/ctransformers-cuBLAS-wheels.*|#https://github.com/jllllll/ctransformers-cuBLAS-wheels|g' \
  "$OOBABOOGA_ROOT_DIR/requirements.txt"

echo "git+https://github.com/oobabooga/torch-grammar.git@main" >> "$OOBABOOGA_ROOT_DIR/requirements.txt"
echo "git+https://github.com/UKPLab/sentence-transformers.git@master" >> "$OOBABOOGA_ROOT_DIR/requirements.txt"

cat $OOBABOOGA_ROOT_DIR/requirements.txt

# Fix https://github.com/oobabooga/text-generation-webui/issues/4644
sed 's|to(self\.projector_device)|to(self\.projector_device,dtype=self\.projector_dtype)|' -i "$OOBABOOGA_ROOT_DIR/extensions/multimodal/pipelines/llava/llava.py" \

# Fix: cannot uninstall 'blinker': It is a distutils installed project
uv pip install --reinstall blinker

# Create a symbolic link from /opt/GPTQ-for-LLaMa/*.py to oobabooga root dir
ln -s /opt/GPTQ-for-LLaMa/*.py "$OOBABOOGA_ROOT_DIR"

# Install text-generation-webui requirements
uv pip install -r "$OOBABOOGA_ROOT_DIR/requirements.txt"

# Install text-generation-webui extensions
cd "$OOBABOOGA_ROOT_DIR"
PYTHONPATH="$OOBABOOGA_ROOT_DIR" python3 -c "from one_click import install_extensions_requirements; install_extensions_requirements()"

# Cleanup
rm -rf \
  /var/lib/apt/lists/* \
  "$OOBABOOGA_ROOT_DIR/api-examples" \
  "$OOBABOOGA_ROOT_DIR/docker" \
  "$OOBABOOGA_ROOT_DIR/docs"

apt-get clean
