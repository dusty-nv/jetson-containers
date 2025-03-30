#!/bin/bash
#==============================================================================
#
#  ██╗███████╗████████╗███████╗ ██████╗ ███╗   ██╗
#  ██║██╔════╝╚══██╔══╝██╔════╝██╔═══██╗████╗  ██║
#  ██║█████╗     ██║   ███████╗██║   ██║██╔██╗ ██║
#  ██║██╔══╝     ██║   ╚════██║██║   ██║██║╚██╗██║
#  ██║███████╗   ██║   ███████║╚██████╔╝██║ ╚████║
#  ╚═╝╚══════╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
#
# Container Build Tool for NVIDIA Jetson ML Development
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
echo -e "${CYAN}${BOLD}"
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║                                                                     ║"
echo "║   ░░░▒▒▒▓▓▓███ JETSON ML CONTAINER BUILDER ███▓▓▓▒▒▒░░░             ║"
echo "║                                                                     ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Get current timestamp
TIMESTAMP=$(date +"%Y-%m-%d-%H%M")
USERNAME=${USER:-"kairin"}
IMAGE_NAME="kairin/001:${TIMESTAMP}-1"

# Display build information
echo -e "${YELLOW}${BOLD}BUILD INFORMATION${RESET}"
echo -e "${BLUE}• Timestamp:${RESET} ${TIMESTAMP}"
echo -e "${BLUE}• Username:${RESET} ${USERNAME}"
echo -e "${BLUE}• Image Name:${RESET} ${IMAGE_NAME}"
echo -e "${BLUE}• Builder:${RESET} mybuilder"
echo -e "${BLUE}• Platform:${RESET} linux/arm64"
echo -e "${BLUE}• Base Image:${RESET} kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu"
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

# Run the build command
docker buildx build \
    --builder mybuilder \
    --platform linux/arm64 \
    -t ${IMAGE_NAME} \
    --build-arg BASE_IMAGE=kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu \
    --build-arg TRITON_VERSION=2.0.0 \
    --build-arg TRITON_BRANCH=main \
    --push \
    .

BUILD_SUCCESS=$?

# Show build result
if [ $BUILD_SUCCESS -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}✅ BUILD SUCCESSFUL!${RESET}"
    echo -e "${GREEN}Image ${IMAGE_NAME} has been built and pushed to registry.${RESET}"
    echo -e "${YELLOW}To verify the image, run:${RESET}"
    echo -e "  ${CYAN}./verify.sh ${IMAGE_NAME}${RESET}"
else
    echo -e "\n${RED}${BOLD}❌ BUILD FAILED!${RESET}"
    echo -e "${RED}Please check the build logs above for errors.${RESET}"
    echo -e "${RED}Make sure you have proper Docker and NVIDIA setup.${RESET}"
fi

echo -e "\n${CYAN}${BOLD}========== BUILD PROCESS COMPLETE ===========${RESET}"
