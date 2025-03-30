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
# NVIDIA Jetson ML Container Builder
# 
# This script builds a Docker container for ML development on NVIDIA Jetson platforms
# with a comprehensive set of libraries and tools pre-installed.
#
# Features:
# - Interactive dialog-based interface
# - Automatic timestamp-based image naming
# - Configurable build options (cache, no-cache)
# - Integration with verify.sh for immediate verification
# - Visual progress indicators and colorful output
#
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

# Check if dialog is installed, install if needed
check_dialog() {
    if ! command -v dialog &> /dev/null; then
        echo -e "${YELLOW}Dialog utility not found. Installing...${RESET}"
        sudo apt-get update
        sudo apt-get install -y dialog
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install dialog. Continuing with basic interface.${RESET}"
            USE_DIALOG=false
            return
        fi
    fi
    USE_DIALOG=true
}

# Get current timestamp and set default values
setup_defaults() {
    TIMESTAMP=$(date +"%Y-%m-%d-%H%M")
    USERNAME=${USER:-"kairin"}
    IMAGE_NAME="kairin/001:${TIMESTAMP}-1"
    BASE_IMAGE="kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu"
    USE_CACHE=true
    AUTO_VERIFY=true
}

# Show build configuration menu with dialog
show_dialog_menu() {
    # Create temporary file for dialog output
    TEMP_FILE=$(mktemp)
    
    # Show dialog form
    dialog --backtitle "NVIDIA Jetson Container Builder" \
           --title "Build Configuration" \
           --form "\nConfigure build parameters:" 20 70 8 \
           "Image Name:" 1 1 "$IMAGE_NAME" 1 18 50 0 \
           "Base Image:" 2 1 "$BASE_IMAGE" 2 18 50 0 \
           2> $TEMP_FILE
    
    # Read form values if dialog was successful
    if [ $? -eq 0 ]; then
        IMAGE_NAME=$(sed -n '1p' $TEMP_FILE)
        BASE_IMAGE=$(sed -n '2p' $TEMP_FILE)
    else
        # User canceled, exit
        rm -f $TEMP_FILE
        echo -e "${YELLOW}Build canceled by user.${RESET}"
        exit 0
    fi
    
    # Show additional options
    dialog --backtitle "NVIDIA Jetson Container Builder" \
           --title "Build Options" \
           --checklist "\nSelect build options:" 15 60 3 \
           "USE_CACHE" "Use Docker build cache" $([ "$USE_CACHE" = true ] && echo "on" || echo "off") \
           "AUTO_VERIFY" "Automatically verify after build" $([ "$AUTO_VERIFY" = true ] && echo "on" || echo "off") \
           2> $TEMP_FILE
    
    # Read checklist values
    if [ $? -eq 0 ]; then
        SELECTED=$(cat $TEMP_FILE)
        if [[ $SELECTED == *"USE_CACHE"* ]]; then
            USE_CACHE=true
        else
            USE_CACHE=false
        fi
        if [[ $SELECTED == *"AUTO_VERIFY"* ]]; then
            AUTO_VERIFY=true
        else
            AUTO_VERIFY=false
        fi
    fi
    
    # Clean up
    rm -f $TEMP_FILE
}

# Show build configuration with basic terminal UI
show_basic_menu() {
    echo -e "${CYAN}${BOLD}NVIDIA Jetson Container Builder${RESET}"
    echo -e "${BLUE}Current build configuration:${RESET}"
    echo -e "  Image Name: ${YELLOW}$IMAGE_NAME${RESET}"
    echo -e "  Base Image: ${YELLOW}$BASE_IMAGE${RESET}"
    echo ""
    
    read -p "Do you want to change the image name? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter new image name: " -r new_image
        if [ -n "$new_image" ]; then
            IMAGE_NAME="$new_image"
        fi
    fi
    
    read -p "Do you want to change the base image? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter new base image: " -r new_base
        if [ -n "$new_base" ]; then
            BASE_IMAGE="$new_base"
        fi
    fi
    
    read -p "Use Docker build cache? (Y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        USE_CACHE=false
    else
        USE_CACHE=true
    fi
    
    read -p "Automatically verify after build? (Y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        AUTO_VERIFY=false
    else
        AUTO_VERIFY=true
    fi
}

# Display build header
show_header() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔═════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                     ║"
    echo "║   ░░░▒▒▒▓▓▓███ JETSON ML CONTAINER BUILDER ███▓▓▓▒▒▒░░░             ║"
    echo "║                                                                     ║"
    echo "╚═════════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

# Show build information
show_build_info() {
    echo -e "${YELLOW}${BOLD}BUILD INFORMATION${RESET}"
    echo -e "${BLUE}• Timestamp:${RESET} ${TIMESTAMP}"
    echo -e "${BLUE}• Username:${RESET} ${USERNAME}"
    echo -e "${BLUE}• Image Name:${RESET} ${IMAGE_NAME}"
    echo -e "${BLUE}• Base Image:${RESET} ${BASE_IMAGE}"
    echo -e "${BLUE}• Builder:${RESET} mybuilder"
    echo -e "${BLUE}• Platform:${RESET} linux/arm64"
    echo -e "${BLUE}• Use Cache:${RESET} $([ "$USE_CACHE" = true ] && echo "Yes" || echo "No")"
    echo -e "${BLUE}• Auto Verify:${RESET} $([ "$AUTO_VERIFY" = true ] && echo "Yes" || echo "No")"
    echo ""
}

# Check and setup buildx
setup_buildx() {
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
}

# Run the build process
run_build() {
    echo -e "\n${YELLOW}${BOLD}STARTING BUILD PROCESS${RESET}"
    echo -e "${CYAN}Building ML container for NVIDIA Jetson...${RESET}"
    echo -e "${CYAN}This may take a while depending on your connection and hardware.${RESET}"
    echo ""
    
    # Prepare build command
    BUILD_CMD="docker buildx build --builder mybuilder --platform linux/arm64"
    
    # Add no-cache flag if needed
    if [ "$USE_CACHE" = false ]; then
        BUILD_CMD="$BUILD_CMD --no-cache"
    fi
    
    # Add image name and build arguments
    BUILD_CMD="$BUILD_CMD -t ${IMAGE_NAME} --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg TRITON_VERSION=2.0.0 --build-arg TRITON_BRANCH=main"
    
    # Add push flag to push to registry
    BUILD_CMD="$BUILD_CMD --push ."
    
    # Execute build command
    eval $BUILD_CMD
    
    # Store build result
    BUILD_SUCCESS=$?
    
    # Show build result
    if [ $BUILD_SUCCESS -eq 0 ]; then
        echo -e "\n${GREEN}${BOLD}✅ BUILD SUCCESSFUL!${RESET}"
        echo -e "${GREEN}Image ${IMAGE_NAME} has been built and pushed to registry.${RESET}"
        
        # Run verification if auto-verify is enabled
        if [ "$AUTO_VERIFY" = true ]; then
            echo -e "\n${YELLOW}${BOLD}STARTING AUTOMATIC VERIFICATION${RESET}"
            echo -e "${CYAN}Running verification script on the new image...${RESET}"
            
            # Check if verify.sh exists and is executable
            if [ -f "./verify.sh" ] && [ -x "./verify.sh" ]; then
                ./verify.sh "$IMAGE_NAME"
            else
                echo -e "${RED}Verification script not found or not executable.${RESET}"
                echo -e "${YELLOW}To verify manually, run: ./verify.sh ${IMAGE_NAME}${RESET}"
            fi
        else
            echo -e "${YELLOW}To verify the image, run:${RESET}"
            echo -e "  ${CYAN}./verify.sh ${IMAGE_NAME}${RESET}"
        fi
    else
        echo -e "\n${RED}${BOLD}❌ BUILD FAILED!${RESET}"
        echo -e "${RED}Please check the build logs above for errors.${RESET}"
        echo -e "${RED}Make sure you have proper Docker and NVIDIA setup.${RESET}"
    fi
    
    echo -e "\n${CYAN}${BOLD}========== BUILD PROCESS COMPLETE ===========${RESET}"
}

# Main function
main() {
    # Check if dialog is available
    check_dialog
    
    # Set up default values
    setup_defaults
    
    # Show header
    show_header
    
    # Show configuration menu
    if [ "$USE_DIALOG" = true ]; then
        show_dialog_menu
    else
        show_basic_menu
    fi
    
    # Show header and build info
    show_header
    show_build_info
    
    # Setup buildx
    setup_buildx
    
    # Run the build
    run_build
}

# Run main function
main
