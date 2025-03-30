#!/bin/bash
#==============================================================================
#
#  ██╗   ██╗███████╗██████╗ ██╗███████╗██╗   ██╗
#  ██║   ██║██╔════╝██╔══██╗██║██╔════╝╚██╗ ██╔╝
#  ██║   ██║█████╗  ██████╔╝██║█████╗   ╚████╔╝ 
#  ╚██╗ ██╔╝██╔══╝  ██╔══██╗██║██╔══╝    ╚██╔╝  
#   ╚████╔╝ ███████╗██║  ██║██║██║        ██║   
#    ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   
#
# Container Verification Tool for NVIDIA Jetson ML Development
# Current Version: 2025-03-30
#==============================================================================
#
# This script verifies the ML container for NVIDIA Jetson platforms.
# It checks:
#
# - GPU access and NVIDIA runtime
# - Python environment and version
# - Core ML frameworks (PyTorch, TensorFlow, JAX)
# - Hugging Face libraries (Transformers, Diffusers)
# - Optimization libraries (Triton, xFormers, etc.)
# - System utilities (FFmpeg, GStreamer)
#
# The script can operate in two modes:
# 1. As a standalone script that runs tests on a specified container image
# 2. Inside a container to verify the container environment itself
#==============================================================================

# Reminder by KAIRIN -- please do not remove the above comments only improve or add on to the information above. This is intended to help remind you what # has been installed, what needs to be installed subsequently and so on. Also please ensure, the latest version compatible for all the apps and software # are installed either by running an update command at the end or something like that. Please indicate what are the changes made with timestamps.

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Detect if running inside a container
INSIDE_CONTAINER=false
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup; then
    INSIDE_CONTAINER=true
fi

# Header
echo -e "${MAGENTA}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║   ░░░▒▒▒▓▓▓███ JETSON ML CONTAINER VERIFIER ███▓▓▓▒▒▒░░░             ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}${BOLD}╔═══ $1 ════════════════════════════════════════════════╗${RESET}"
}

# Function to run verification inside container
verify_inside_container() {
    echo -e "${CYAN}${BOLD}VERIFYING CURRENT CONTAINER ENVIRONMENT${RESET}\n"
    
    # Verify Python
    print_section "PYTHON ENVIRONMENT"
    python3 --version
    echo ""
    
    # Verify key Python packages
    print_section "CORE ML FRAMEWORKS"
    PACKAGES=(
        "numpy" 
        "torch" 
        "tensorflow" 
        "jax" 
        "transformers"
        "diffusers"
        "onnx"
    )
    
    for package in "${PACKAGES[@]}"; do
        echo -ne "${CYAN}Checking ${package}...${RESET} "
        if python3 -c "import $package; print(getattr($package, '__version__', 'installed'))" 2>/dev/null; then
            echo ""
        else
            echo -e "${RED}❌ Not installed or error importing${RESET}"
        fi
    done
    
    # Verify optimization libraries
    print_section "OPTIMIZATION LIBRARIES"
    OPT_PACKAGES=(
        "triton"
        "bitsandbytes"
        "xformers"
    )
    
    for package in "${OPT_PACKAGES[@]}"; do
        echo -ne "${CYAN}Checking ${package}...${RESET} "
        if python3 -c "import $package; print('✅ Installed')" 2>/dev/null; then
            echo ""
        else
            echo -e "${YELLOW}⚠️ Not installed or error importing${RESET}"
        fi
    done
    
    # Verify system utilities
    print_section "SYSTEM UTILITIES"
    UTILS=("ffmpeg -version" "pkg-config --version")
    UTIL_NAMES=("FFmpeg" "pkg-config")
    
    for i in "${!UTILS[@]}"; do
        echo -ne "${CYAN}Checking ${UTIL_NAMES[$i]}...${RESET} "
        if bash -c "${UTILS[$i]}" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Installed${RESET}"
        else
            echo -e "${YELLOW}⚠️ Not installed or not working${RESET}"
        fi
    done
    
    # Verify NVIDIA GPU access
    print_section "NVIDIA GPU ACCESS"
    if nvidia-smi >/dev/null
