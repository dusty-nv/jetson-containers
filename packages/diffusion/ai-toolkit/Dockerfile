#--- 
# name: ai-toolkit 
# group: diffusion 
# depends: [pytorch, torchvision, torchaudio, transformers, bitsandbytes, opencv, huggingface_hub] 
# requires: '>=34.1.0' 
# docs: docs.md 
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

# Clona el repositorio de Diffusers
RUN git clone https://github.com/huggingface/diffusers.git && \
    cd diffusers && \
    # Filtra torch y torchvision del setup.py para evitar la reinstalación
    sed -i '/torch/d' setup.py && \
    sed -i '/torchvision/d' setup.py && \
    # Instala Diffusers sin reinstalar torch y torchvision
    pip3 install --no-cache-dir .

# Instala las dependencias adicionales para ai-toolkit
COPY requirements.txt /opt/ai-toolkit/
RUN grep -v -E 'torch|torchvision' /opt/ai-toolkit/requirements.txt > temp_requirements.txt && \
    pip3 install --no-cache-dir -r temp_requirements.txt && \
    pip3 install --force-reinstall 'numpy<2'

# Configura el directorio de trabajo
WORKDIR /opt/ai-toolkit/

CMD ["python3", "main.py"]
    