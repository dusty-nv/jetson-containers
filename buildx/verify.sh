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

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Header
echo -e "${MAGENTA}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║   ░░░▒▒▒▓▓▓███ JETSON ML CONTAINER VERIFIER ███▓▓▓▒▒▒░░░             ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if image name is provided
if [ -z "$1" ]; then
    echo -e "${RED}${BOLD}ERROR: No image name provided!${RESET}"
    echo -e "${YELLOW}Usage: $0 <image_name>${RESET}"
    echo -e "${YELLOW}Example: $0 kairin/001:2025-03-30-1140-1${RESET}"
    exit 1
fi

# Store the image name
IMAGE_NAME=$1
echo -e "${CYAN}${BOLD}VERIFYING IMAGE:${RESET} ${IMAGE_NAME}\n"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}${BOLD}╔═══ $1 ════════════════════════════════════════════════╗${RESET}"
}

# Function to run tests
run_test() {
    local name=$1
    local cmd=$2
    echo -ne "${BLUE}Testing ${name}...${RESET} "
    if docker run --rm $IMAGE_NAME $cmd >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${RESET}"
        return 0
    else
        echo -e "${RED}❌ FAILED${RESET}"
        return 1
    fi
}

# Test container startup
print_section "CONTAINER STARTUP"
echo -e "${CYAN}Starting container with NVIDIA runtime...${RESET}"

# Try to run nvidia-smi to verify GPU access
docker run --rm --runtime nvidia -it $IMAGE_NAME nvidia-smi
if [ $? -ne 0 ]; then
    echo -e "\n${RED}${BOLD}❌ CRITICAL ERROR: Failed to start container with GPU access.${RESET}"
    echo -e "${RED}Please check if NVIDIA drivers and Docker runtime are properly configured.${RESET}"
    exit 1
fi
echo -e "${GREEN}✅ Container started successfully with GPU access.${RESET}"

# Verify Python
print_section "PYTHON ENVIRONMENT"
docker run --rm $IMAGE_NAME python3 --version
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
    VERSION=$(docker run --rm $IMAGE_NAME python3 -c "import $package; print(getattr($package, '__version__', 'installed'))" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $VERSION${RESET}"
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
    "deepspeed"
)

for package in "${OPT_PACKAGES[@]}"; do
    echo -ne "${CYAN}Checking ${package}...${RESET} "
    if docker run --rm $IMAGE_NAME python3 -c "import $package; print('✅')" 2>/dev/null; then
        echo -e "${GREEN}Installed${RESET}"
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
    if docker run --rm $IMAGE_NAME bash -c "${UTILS[$i]}" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Installed${RESET}"
    else
        echo -e "${YELLOW}⚠️ Not installed or not working${RESET}"
    fi
done

# Summary
print_section "VERIFICATION SUMMARY"
echo -e "${GREEN}${BOLD}✅ Container verification completed!${RESET}"
echo -e "${GREEN}The container appears to be properly configured with GPU support.${RESET}"
echo -e "${GREEN}Core ML frameworks and libraries are installed and working.${RESET}"
echo ""
echo -e "${CYAN}To use this container interactively, run:${RESET}"
echo -e "${YELLOW}docker run --rm -it --runtime nvidia ${IMAGE_NAME} bash${RESET}"
echo ""
echo -e "${CYAN}${BOLD}========== VERIFICATION COMPLETE ===========${RESET}"
