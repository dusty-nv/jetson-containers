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
${GREEN}${BOLD}
        _______   ________   __________   ______    ________    _______  
      /"      \ /"       ) ("     _   ") /    " \  /"       )  /"      \ 
     |:        (:   \___/   )__/  \\__/ // ____  \(:   \___/  |:        |
     |_____/   )\___  \        \\_ /   /  /    ) :)\___  \    |_____/   )
      //      /  __/  \\       |.  |  (: (____/ //  __/  \\    //      / 
     |:  __   \ /" \   :)      \:  |   \        /  /" \   :)  |:  __   \ 
     |__|  \___(_______/        \__|    \"_____/  (_______/   |__|  \___)
                                                                        
     ________    __    __  _______   _______   ________  _______         
    |"      "\  /""\  |" \|"     "\ /"     "| /"       )/"     "|        
    (.  ___  :)/    \ ||  \:. ______):      |(: ______/(:      |_        
    |: \   ) |/' /\  \|:  ||  ____/  |:     |/ \/    |  |:      "|       
    (| (___\ ://  __  \  / : \        |.  ___|// _____)  |.      :|      
    |:       /   /  \   \ |_/ \       |: |    (:  (      |:       |      
    (________/__/    \___\)__) \      |__|     \__/       \______/       
${RESET}
EOF

echo
echo -e "${MAGENTA}${BOLD}===================== JETSON AI CONTAINER LAUNCHER =====================${RESET}"
echo

# Function to show help message
show_help() {
    echo -e "${YELLOW}${BOLD}Usage:${RESET} $0 [OPTIONS]"
    echo
    echo -e "${CYAN}Options:${RESET}"
    echo -e "  ${GREEN}-h, --help${RESET}       Show this help message"
    echo -e "  ${GREEN}-t, --tag TAG${RESET}    Specify the Docker image tag (default: latest)"
    echo -e "  ${GREEN}-v, --volume DIR${RESET} Mount a volume (can be used multiple times)"
    echo -e "  ${GREEN}-p, --port PORT${RESET}  Publish a port (can be used multiple times)"
    echo -e "  ${GREEN}-g, --gpu${RESET}        Enable GPU support"
    echo -e "  ${GREEN}-d, --detach${RESET}     Run container in background"
    echo
    echo -e "${YELLOW}${BOLD}Examples:${RESET}"
    echo -e "  $0 -g -t 2025-03-30 -v /data:/workspace/data"
    echo -e "  $0 -g -p 8888:8888 -p 6006:6006 -v /models:/workspace/models"
    echo
}

# Default values
TAG="latest"
VOLUMES=()
PORTS=()
GPU_FLAG=""
DETACH_FLAG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--tag)
            TAG="$2"
            shift
            shift
            ;;
        -v|--volume)
            VOLUMES+=("$2")
            shift
            shift
            ;;
        -p|--port)
            PORTS+=("$2")
            shift
            shift
            ;;
        -g|--gpu)
            GPU_FLAG="--gpus all"
            shift
            ;;
        -d|--detach)
            DETACH_FLAG="-d"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${RESET}"
            show_help
            exit 1
            ;;
    esac
done

# Construct volume mounts
VOLUME_ARGS=""
for vol in "${VOLUMES[@]}"; do
    VOLUME_ARGS="$VOLUME_ARGS -v $vol"
done

# Construct port mappings
PORT_ARGS=""
for port in "${PORTS[@]}"; do
    PORT_ARGS="$PORT_ARGS -p $port"
done

# Display run information
echo -e "${YELLOW}${BOLD}CONTAINER CONFIGURATION:${RESET}"
echo -e "${CYAN}Image:${RESET} kairin/001:$TAG"
echo -e "${CYAN}GPU Support:${RESET} ${GPU_FLAG:+Enabled}"
echo -e "${CYAN}Detached Mode:${RESET} ${DETACH_FLAG:+Enabled}"

if [ ${#VOLUMES[@]} -gt 0 ]; then
    echo -e "${CYAN}Volumes:${RESET}"
    for vol in "${VOLUMES[@]}"; do
        echo -e "  - $vol"
    done
fi

if [ ${#PORTS[@]} -gt 0 ]; then
    echo -e "${CYAN}Ports:${RESET}"
    for port in "${PORTS[@]}"; do
        echo -e "  - $port"
    done
fi

echo

# Confirmation
read -p "$(echo -e ${YELLOW}${BOLD}"Do you want to launch the container with these settings? (y/n): "${RESET})" confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo -e "${RED}Operation cancelled.${RESET}"
    exit 0
fi

echo -e "${GREEN}${BOLD}Launching container...${RESET}"

# Run the container
docker run $DETACH_FLAG \
    $GPU_FLAG \
    $VOLUME_ARGS \
    $PORT_ARGS \
    -it \
    --name jetson-ai-$(date +%s) \
    kairin/001:$TAG

echo -e "${GREEN}${BOLD}Container launched successfully!${RESET}"
