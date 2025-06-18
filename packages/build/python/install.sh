#!/usr/bin/env bash
# Python installer using Micromamba
set -x

curl -Ls https://micro.mamba.pm/api/micromamba/linux-$(uname -m | sed 's/x86_64/64/;s/aarch64/aarch64/')/latest | tar -xvj bin/micromamba
mv bin/micromamba /usr/local/bin/

export MAMBA_ROOT_PREFIX=/opt/conda
export CONDA_PREFIX=/opt/conda
export PATH=/opt/conda/bin:$PATH
mkdir -p $MAMBA_ROOT_PREFIX
eval "$(micromamba shell hook -s posix)"
micromamba shell init -s bash -r $MAMBA_ROOT_PREFIX

# Install Python and core packages in base environment
micromamba install -y -n base \
    python=${PYTHON_VERSION} \
    setuptools \
    packaging \
    "cython<3" \
    wheel \
    pip \
    uv \
    twine \
    psutil \
    pkginfo

# Add micromamba initialization to /etc/profile.d for all shells
cat > /etc/profile.d/mamba.sh << 'EOF'
export MAMBA_EXE=/usr/bin/micromamba
export MAMBA_ROOT_PREFIX=/opt/conda
export PATH=/opt/conda/bin:$PATH

if [ -f "${MAMBA_EXE}" ]; then
    __mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__mamba_setup"
    else
        alias micromamba="$MAMBA_EXE"
    fi
    unset __mamba_setup
fi

micromamba activate
EOF

chmod +x /etc/profile.d/mamba.sh
source /etc/profile.d/mamba.sh

ln -sf /opt/conda/bin/python /usr/local/bin/python3
ln -sf /opt/conda/bin/python /usr/local/bin/python
ln -sf /opt/conda/bin/pip /usr/local/bin/pip3
ln -sf /opt/conda/bin/pip /usr/local/bin/pip

# Verify installation
which python3
python3 --version
which pip3
pip3 --version
