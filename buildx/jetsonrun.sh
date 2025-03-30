#!/bin/bash
#==============================================================================
#
#  ░░░▒▒▒▓▓▓███ JETSON ML CONTAINER BUILDER ███▓▓▓▒▒▒░░░             
#
# Container Build Tool for NVIDIA Jetson ML Development
# Current Version: 2025-03-30
#==============================================================================
# 
# This script builds a comprehensive machine learning container for NVIDIA Jetson platforms.
# It includes:
# 
# - System Dependencies and Basic Utilities:
#   build-essential, cmake, git, curl, wget, and more
#
# - Python Core Dependencies: 
#   numpy, scipy, matplotlib, pandas, jupyterlab
#
# - Core ML/DL Framework Layer:
#   PyTorch (from base image), TensorFlow, JAX, Flax, Optax
#
# - Hugging Face Ecosystem:
#   Transformers, Diffusers, Datasets, Tokenizers
#
# - ONNX Runtime & Conversion:
#   ONNX, ONNX Runtime
#
# - OpenAI Triton Compilation:
#   Triton compiler for GPU kernels
#
# - Optimization Libraries:
#   bitsandbytes, flash-attention, xformers
#
# - Computer Vision Libraries:
#   OpenCV
#
# - Media Processing:
#   FFmpeg, GStreamer
#
# - CUDA Extensions:
#   CuPy
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

# Default values
BASE_IMAGE="kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu"
USE_CACHE="yes"
AUTO_VERIFY="no"
TIMESTAMP=$(date +"%Y-%m-%d-%H%M")
USERNAME=${USER:-"kairin"}
IMAGE_NAME="kairin/001:${TIMESTAMP}-1"

# Check if dialog is installed, if not, install it
if ! command -v dialog &> /dev/null; then
    echo -e "${YELLOW}Dialog not found. Installing...${RESET}"
    sudo apt-get update && sudo apt-get install -y dialog
fi

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}${BOLD}ERROR: Docker is not installed. Please install Docker first.${RESET}"
    exit 1
fi

# Function to display ASCII art header in terminal (non-dialog mode)
show_header() {
    echo -e "${CYAN}${BOLD}"
    echo "╔═════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                     ║"
    echo "║   ░░░▒▒▒▓▓▓███ JETSON ML CONTAINER BUILDER ███▓▓▓▒▒▒░░░             ║"
    echo "║                                                                     ║"
    echo "╚═════════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

# Show the welcome dialog
dialog --clear --title "Jetson ML Container Builder" \
       --backtitle "NVIDIA Jetson ML Development" \
       --colors \
       --msgbox "\Z1\Zb╔══════════════════════════════════════════════════════════════════════════════╗\Zn
\Z1\Zb║                                                                                      ║\Zn
\Z1\Zb║                      JETSON ML CONTAINER BUILDER                                     ║\Zn
\Z1\Zb║                                                                                      ║\Zn
\Z1\Zb╚════════════════════════════════════════════════════════════════════════════════════╚╝\Zn

This tool builds a comprehensive machine learning container 
optimized for NVIDIA Jetson platforms.

The container includes:
• PyTorch, TensorFlow, JAX ecosystems
• Hugging Face Transformers, Diffusers
• ONNX Runtime and optimization libraries
• Triton compiler for GPU kernels
• Media libraries (OpenCV, FFmpeg, GStreamer)
• CUDA extensions and optimization tools

Press OK to configure your build options." 30 80

# Main configuration menu
while true; do
    OPTIONS=$(dialog --clear --title "Build Configuration" \
            --backtitle "NVIDIA Jetson ML Development" \
            --ok-label "Build" \
            --cancel-label "Exit" \
            --extra-button \
            --extra-label "Advanced" \
            --colors \
            --form "\Z4\ZbBuild Configuration\Zn" 20 80 0 \
            "Image Name:" 1 1 "$IMAGE_NAME" 1 20 45 0 \
            "Base Image:" 2 1 "$BASE_IMAGE" 2 20 45 0 \
            "Use Cache:" 3 1 "$USE_CACHE" 3 20 5 0 \
            "Auto Verify:" 4 1 "$AUTO_VERIFY" 4 20 5 0 \
            2>&1 >/dev/tty)
    
    BUTTON=$?
    
    case $BUTTON in
        0) # Build button pressed
            IMAGE_NAME=$(echo "$OPTIONS" | sed -n 1p)
            BASE_IMAGE=$(echo "$OPTIONS" | sed -n 2p)
            USE_CACHE=$(echo "$OPTIONS" | sed -n 3p)
            AUTO_VERIFY=$(echo "$OPTIONS" | sed -n 4p)
            break
            ;;
        1) # Exit button pressed
            clear
            echo -e "${YELLOW}Build canceled. Exiting.${RESET}"
            exit 0
            ;;
        3) # Advanced button pressed
            ADVANCED=$(dialog --clear --title "Advanced Configuration" \
                    --backtitle "NVIDIA Jetson ML Development" \
                    --ok-label "Apply" \
                    --cancel-label "Back" \
                    --checklist "Select components to include:" 25 80 15 \
                    "PYTORCH" "PyTorch and TorchVision" on \
                    "TENSORFLOW" "TensorFlow" on \
                    "JAX" "JAX, Flax, and Optax" on \
                    "TRANSFORMERS" "Hugging Face Transformers" on \
                    "DIFFUSERS" "Hugging Face Diffusers" on \
                    "ONNX" "ONNX Runtime" on \
                    "TRITON" "OpenAI Triton" on \
                    "XFORMERS" "xFormers optimization library" on \
                    "OPENCV" "OpenCV" on \
                    "FFMPEG" "FFmpeg" on \
                    "GSTREAMER" "GStreamer" on \
                    "CUPY" "CuPy (CUDA NumPy)" on \
                    2>&1 >/dev/tty)
                    
            # We would parse the selections here, but for now we'll just continue
            # with all components enabled by default
            ;;
    esac
done

# Clear the screen for build output
clear
show_header

# Display build information
echo -e "${YELLOW}${BOLD}BUILD INFORMATION${RESET}"
echo -e "${BLUE}• Image Name:${RESET} ${IMAGE_NAME}"
echo -e "${BLUE}• Base Image:${RESET} ${BASE_IMAGE}"
echo -e "${BLUE}• Use Cache:${RESET} ${USE_CACHE}"
echo -e "${BLUE}• Auto Verify:${RESET} ${AUTO_VERIFY}"
echo -e "${BLUE}• Builder:${RESET} mybuilder"
echo -e "${BLUE}• Platform:${RESET} linux/arm64"
echo ""

# Check and setup buildx
echo -e "${YELLOW}${BOLD}BUILDER SETUP${RESET}"
if ! docker buildx ls | grep -q mybuilder; then
    echo -e "${MAGENTA}Creating new builder instance 'mybuilder'...${RESET}"
    docker buildx create --name mybuilder --use
    if [ $? -ne 0 ]; then
        echo -e "${RED}${BOLD}Failed to create builder! Exiting.${RESET}"
        exit 1
    fi
else
    echo -e "${GREEN}Builder 'mybuilder' already exists. Using it.${RESET}"
    docker buildx use mybuilder
fi

# Start build process
echo -e "\n${YELLOW}${BOLD}STARTING BUILD PROCESS${RESET}"
echo -e "${CYAN}Building ML container for NVIDIA Jetson...${RESET}"
echo -e "${CYAN}This may take a while depending on your connection and hardware.${RESET}"
echo ""

# Set cache option
if [ "$USE_CACHE" = "no"; then
    CACHE_OPTION="--no-cache"
else
    CACHE_OPTION=""
fi

# Make a backup copy of verify.sh to be included in the container
cp verify.sh verify.sh.copy

# Add verify.sh to be included in the container
cat > Dockerfile.build <<EOF
# Include the verification script in the container
COPY verify.sh.copy /workspace/verify.sh
RUN chmod +x /workspace/verify.sh
EOF

cat Dockerfile Dockerfile.build > Dockerfile.tmp

# Run the build command
docker buildx build \
    --builder mybuilder \
    --platform linux/arm64 \
    -t ${IMAGE_NAME} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg TRITON_VERSION=2.0.0 \
    --build-arg TRITON_BRANCH=main \
    ${CACHE_OPTION} \
    --push \
    -f Dockerfile.tmp \
    .

BUILD_SUCCESS=$?

# Clean up temporary files
rm -f Dockerfile.tmp Dockerfile.build verify.sh.copy

# Show build result
if [ $BUILD_SUCCESS -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}✅ BUILD SUCCESSFUL!${RESET}"
    echo -e "${GREEN}Image ${IMAGE_NAME} has been built and pushed to registry.${RESET}"
    
    if [ "$AUTO_VERIFY" = "yes" ]; then
        echo -e "\n${YELLOW}${BOLD}AUTOMATICALLY STARTING VERIFICATION...${RESET}"
        ./verify.sh ${IMAGE_NAME}
    else
        echo -e "\n${YELLOW}To verify the image, run:${RESET}"
        echo -e "  ${CYAN}./verify.sh ${IMAGE_NAME}${RESET}"
        echo -e "\n${YELLOW}To run the container interactively:${RESET}"
        echo -e "  ${CYAN}docker run --rm -it --runtime nvidia ${IMAGE_NAME} bash${RESET}"
        echo -e "\n${YELLOW}To verify from inside the container, run:${RESET}"
        echo -e "  ${CYAN}/workspace/verify.sh${RESET}"
    fi
else
    echo -e "\n${RED}${BOLD}❌ BUILD FAILED!${RESET}"
    echo -e "${RED}Please check the build logs above for errors.${RESET}"
    echo -e "${RED}Make sure you have proper Docker and NVIDIA setup.${RESET}"
fi

echo -e "\n${CYAN}${BOLD}========== BUILD PROCESS COMPLETE ===========${RESET}"

# Note: It is important and necessary to keep the dialog box size larger to fit the full length of the Docker file name for future updates.
