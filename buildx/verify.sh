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
# 
# This script verifies a Docker container for ML development on NVIDIA Jetson platforms,
# checking for proper setup of ML libraries, CUDA, GPU access, and other tools.
#
# Features:
# - Can run both outside the container to verify a built image
# - Can run inside the container to verify the current environment
# - Tests GPU access and functionality
# - Verifies core ML frameworks and libraries
# - Checks optimization libraries
# - Tests system utilities
# - Provides colorful, readable output with status icons
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

# Determine if we're running inside or outside the container
INSIDE_CONTAINER=false
if [ -f /.dockerenv ] || grep -q 'docker\|lxc' /proc/1/cgroup; then
    INSIDE_CONTAINER=true
fi

# Header
show_header() {
    echo -e "${MAGENTA}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║   ░░░▒▒▒▓▓▓███ JETSON ML CONTAINER VERIFIER ███▓▓▓▒▒▒░░░             ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}${BOLD}╔═══ $1 ════════════════════════════════════════════════╗${RESET}"
}

# Function to run tests
run_test() {
    local name=$1
    local cmd=$2
    echo -ne "${BLUE}Testing ${name}...${RESET} "
    if $cmd >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${RESET}"
        return 0
    else
        echo -e "${RED}❌ FAILED${RESET}"
        return 1
    fi
}

# Function to run a test inside a container
run_container_test() {
    local name=$1
    local cmd=$2
    local image=$3
    
    echo -ne "${BLUE}Testing ${name}...${RESET} "
    if docker run --rm --runtime nvidia $image bash -c "$cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${RESET}"
        return 0
    else
        echo -e "${RED}❌ FAILED${RESET}"
        return 1
    fi
}

# Function to check Python package
check_package() {
    local package=$1
    local image=$2
    
    echo -ne "${CYAN}Checking ${package}...${RESET} "
    
    if [ "$INSIDE_CONTAINER" = true ]; then
        VERSION=$(python3 -c "import $package; print(getattr($package, '__version__', 'installed'))" 2>/dev/null)
    else
        VERSION=$(docker run --rm $image python3 -c "import $package; print(getattr($package, '__version__', 'installed'))" 2>/dev/null)
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $VERSION${RESET}"
        return 0
    else
        echo -e "${RED}❌ Not installed or error importing${RESET}"
        return 1
    fi
}

# Check if image name is provided when running outside container
check_args() {
    if [ "$INSIDE_CONTAINER" = false ] && [ -z "$1" ]; then
        echo -e "${RED}${BOLD}ERROR: No image name provided!${RESET}"
        echo -e "${YELLOW}Usage: $0 <image_name>${RESET}"
        echo -e "${YELLOW}Example: $0 kairin/001:2025-03-30-1242-1${RESET}"
        echo -e "${YELLOW}Or use '${0} --run' to start and enter the container for verification${RESET}"
        exit 1
    fi
}

# Option to run the container and execute verification inside
run_container() {
    local image=$1
    
    echo -e "${CYAN}${BOLD}Starting container ${image} for interactive verification...${RESET}"
    echo -e "${YELLOW}The verification script will run automatically inside the container.${RESET}"
    echo -e "${YELLOW}You will be dropped to a shell after verification completes.${RESET}"
    echo ""
    
    # Copy the script to a temporary file
    TEMP_SCRIPT=$(mktemp)
    cat $0 > $TEMP_SCRIPT
    chmod +x $TEMP_SCRIPT
    
    # Run the container with the script mounted and set to execute on start
    docker run --rm -it --runtime nvidia \
        -v $TEMP_SCRIPT:/verify.sh \
        $image bash -c "/verify.sh && echo '' && bash"
    
    # Clean up
    rm -f $TEMP_SCRIPT
    exit 0
}

# Test container startup with GPU access
test_gpu() {
    local image=$1
    
    print_section "CONTAINER STARTUP"
    
    if [ "$INSIDE_CONTAINER" = true ]; then
        echo -e "${CYAN}Testing GPU access inside container...${RESET}"
        nvidia-smi
        if [ $? -ne 0 ]; then
            echo -e "\n${RED}${BOLD}❌ CRITICAL ERROR: No GPU access inside container.${RESET}"
            echo -e "${RED}Please check if NVIDIA drivers and Docker runtime are properly configured.${RESET}"
            return 1
        fi
    else
        echo -e "${CYAN}Starting container with NVIDIA runtime...${RESET}"
        docker run --rm --runtime nvidia -it $image nvidia-smi
        if [ $? -ne 0 ]; then
            echo -e "\n${RED}${BOLD}❌ CRITICAL ERROR: Failed to start container with GPU access.${RESET}"
            echo -e "${RED}Please check if NVIDIA drivers and Docker runtime are properly configured.${RESET}"
            return 1
        fi
    fi
    
    echo -e "${GREEN}✅ Container started successfully with GPU access.${RESET}"
    return 0
}

# Verify Python version
verify_python() {
    local image=$1
    
    print_section "PYTHON ENVIRONMENT"
    
    if [ "$INSIDE_CONTAINER" = true ]; then
        python3 --version
    else
        docker run --rm $image python3 
