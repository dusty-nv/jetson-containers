#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Clear the screen
clear

# ASCII Art Header
cat << "EOF"
${CYAN}${BOLD}
      ___      ___________    _________    ________    ____  ___    ___      ___
     |"  \    ("     _   ")  ("     _  ")/("       )  (\"   \|"  \  |"  \    /"  |
     ||   |    )__/  \\__/    )__/  \\__(:   \___/   |.\\   \    |  \   \  //   |
     |:|   |      \\_ /          \\_ /    \___  \    |: \.   \\  |   \\  \/. ./  
     |.|   |      |.  |          |.  |     __/  \\   |.  \    \. |    \.    //   
     |:    |      \:  |          \:  |    /" \   :)  |    \    \ |     \\   /    
     |____|        \__|           \__|   (_______/    \___|\____\)      \__/     
                                                                               
      _   __  ________  ______   ________  ________  ________  ______   
     | \ |" \|"      "\|    " \ |"      "\/"       )/"       )/" _  "\  
     ||  ||  (.  ___  :)\____) :(.  ___  :\(     _/(:   \___/(: ( \___)
     |:  |:  |: \   ) || |". ./|: \   ) :|.\____\   \___  \   \/ \     
     |.  |.  (| (___\ || o \:: (| (___\ |:|___  \\  __/  \\  //  \ _   
     /\  /\  |:       :)|: n    |:       :    \  \ /" \   :)(:   _) \  
    (__\/__)(_______/ |_|      (_______/      \__(_______/  \_______)  
${RESET}
EOF

echo
echo -e "${MAGENTA}${BOLD}===================== JETSON AI CONTAINER VERIFICATION =====================${RESET}"
echo

# System information
echo -e "${YELLOW}${BOLD}SYSTEM INFORMATION:${RESET}"
echo -e "${CYAN}Date & Time:${RESET} $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${CYAN}Hostname:${RESET} $(hostname)"
echo -e "${CYAN}Container ID:${RESET} $(hostname -I | awk '{print $1}')"
echo

# GPU information
echo -e "${YELLOW}${BOLD}NVIDIA GPU INFORMATION:${RESET}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo -e "${RED}nvidia-smi not found. GPU might not be properly configured.${RESET}"
fi
echo

# Check CUDA availability
echo -e "${YELLOW}${BOLD}CHECKING CUDA AVAILABILITY:${RESET}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo

# List installed Python packages
echo -e "${YELLOW}${BOLD}INSTALLED PYTHON PACKAGES:${RESET}"
pip3 list | grep -E 'torch|tensorflow|jax|onnx|transformers|diffusers|triton|bitsandbytes|xformers|deepspeed|flash-attention'
echo

# Check for specific ML libraries
echo -e "${YELLOW}${BOLD}VERIFYING ML LIBRARIES:${RESET}"

check_package() {
    if pip3 list | grep -q $1; then
        echo -e "  ${GREEN}✓${RESET} $1 is installed"
    else
        echo -e "  ${RED}✗${RESET} $1 is not installed"
    fi
}

check_package "torch"
check_package "tensorflow"
check_package "transformers"
check_package "diffusers"
check_package "onnx"
check_package "jax"
check_package "triton"
check_package "bitsandbytes"
check_package "xformers"
check_package "deepspeed"
check_package "opencv-python"
check_package "cupy"

echo
echo -e "${MAGENTA}${BOLD}===================== VERIFICATION COMPLETE =====================${RESET}"
echo
