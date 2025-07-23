#!/bin/bash
#set -x # Enable debug mode

################################################################################
# JETSON CONTAINER DOCKER CONFIGURATION SCRIPT
# 
# This script handles all Docker-related configuration for Jetson devices:
# 1. Installing Docker with NVIDIA container runtime if needed
# 2. Configuring Docker with NVIDIA runtime as default
# 3. Migrating Docker data directory to NVMe SSD for better performance
# 4. Managing docker group membership for current user
#
# Usage:
#   ./configure-docker.sh [OPTIONS]
#   
# Options:
#   --help             Show help message
#   --test=<function>  Test a specific function
#   --no-migrate      Skip Docker data directory migration
################################################################################

# Pretty print functions
print_section() {
    local message=$1
    echo -e "\n\033[48;5;130m\033[97m >>> $message \033[0m"
}

log() {
    local level=$1
    local message=$2
    case "$level" in
        INFO)  echo -e "\033[36m[INFO]\033[0m $message" ;;      # Cyan text
        WARN)  echo -e "\033[38;5;205m[WARN]\033[0m $message" ;; # Magenta text
        ERROR) echo -e "\033[1;31m[ERROR]\033[0m $message" ;;   # Red text (bold)
        *)     echo "[LOG]  $message" ;;                        # Default log
    esac
}

################################################################################
# DOCKER INSTALLATION AND CONFIGURATION
################################################################################

check_docker_is_installed() {
    if command -v docker &> /dev/null; then
        log INFO "✅ Docker is already installed."
        return 0
    else
        log WARN "❌ Docker is NOT installed."
        return 1
    fi
}

install_docker() {
    log INFO "⚠️ Docker and NVIDIA container runtime will be installed."
    log INFO "⚠️ This process requires internet access and may take a few minutes."
    echo
    # Ask for user confirmation
    read -p "Are you sure you want to install Docker? (y/N): " confirm
    confirm=${confirm,,}  # Convert to lowercase
    if [[ "$confirm" != "y" && "$confirm" != "yes" ]]; then
        echo "❌ Installation aborted by user."
        return 1
    fi
    # Proceed with installation
    log INFO "✅ Proceeding with Docker installation..."
    cd /tmp/
    if [ -d "install-docker" ]; then
        rm -rf install-docker
    fi
    git clone https://github.com/jetsonhacks/install-docker.git
    cd install-docker
    bash ./install_nvidia_docker.sh
    log INFO "✅ Docker and NVIDIA runtime installation complete!"
}

# Add user to docker group
setup_docker_group() {
    # Check if user is in docker group
    if groups $USER | grep -q '\bdocker\b'; then
        log INFO "✅ User '$USER' is already in the docker group."
    else
        log INFO "Adding user to docker group..."
        sudo usermod -aG docker $USER
        log WARN "⚠️ Please log out and log back in for docker group changes to take effect."
        log WARN "   Alternatively, run: newgrp docker"
    fi
}

# Configure Docker with NVIDIA runtime as default
configure_docker_runtime() {
    if grep -q '"default-runtime": "nvidia"' /etc/docker/daemon.json 2>/dev/null; then
        log INFO "✅ NVIDIA runtime is already set as default in Docker daemon.json"
    else
        log WARN "❌ NVIDIA runtime is NOT set as default - configuring now..."
        
        # Check if jq is installed
        if ! command -v jq &> /dev/null; then
            log INFO "Installing jq for JSON processing..."
            sudo apt install -y jq
        fi
        
        # Configure NVIDIA runtime as default
        if [ -f "/etc/docker/daemon.json" ]; then
            sudo jq '. + {"default-runtime": "nvidia"}' /etc/docker/daemon.json | \
                sudo tee /etc/docker/daemon.json.tmp > /dev/null && \
                sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
        else
            echo '{"default-runtime": "nvidia"}' | sudo tee /etc/docker/daemon.json > /dev/null
        fi
        
        # Restart Docker to apply changes
        log INFO "Restarting Docker service to apply changes..."
        sudo systemctl restart docker
        log INFO "✅ NVIDIA runtime is now set as default."
    fi
}

check_l4t_installed_on_nvme() {
    root_device=$(findmnt -n -o SOURCE /)
    if [[ "$root_device" =~ nvme ]]; then
        return 0
    fi
    return 1
}

setup_docker() {
    if ! check_docker_is_installed; then
        install_docker
    fi
    
    # Configure Docker runtime
    configure_docker_runtime
    
    # Ensure user is in docker group
    setup_docker_group
    
    # Skip data-root migration if L4T is installed on NVMe
    if check_l4t_installed_on_nvme; then
        log WARN "✅ No need to migrate Docker data as your entire system is on NVMe SSD"
        return 0
    fi
    
    # Skip data-root migration if it's already configured
    if grep -q '"data-root"' /etc/docker/daemon.json 2>/dev/null; then
        log INFO "✅ Docker data-root dir is already set"
        return 0
    fi
    
    # L4T is installed on eMMC and Docker data-root to be migrated to SSD
    log INFO "List all NVMe mount points"
    nvme_mount_points=($(lsblk -nr -o NAME,MOUNTPOINT | awk '/^nvme/ && $2 != "" {print $2}'))
    if [[ ${#nvme_mount_points[@]} -eq 0 ]]; then
        echo "❌ No NVMe mount points found. Exiting."
        exit 1
    fi
    
    # List all NVMe mount points
    for mp in "${nvme_mount_points[@]}"; do
        echo "Mounted NVMe: $mp"
    done
    
    # Set default to the first NVMe mount point
    default_mount_point_docker_root_dir="${nvme_mount_points[0]}/docker"
    echo -n "Enter the Docker data-root dir [Default: $default_mount_point_docker_root_dir]: "
    read -e -i "$default_mount_point_docker_root_dir" selected_mount_point_docker_root_dir
    
    # Migrate Docker data-root directory to SSD
    log INFO "Migrate Docker directory to SSD"
    sudo systemctl stop docker
    log INFO "Moving the existing Docker data directory (/var/lib/docker/)..."
    sudo du -csh /var/lib/docker/
    log WARN "⚠️ This may take time if you were operating on eMMC/SD"
    sudo mkdir -p "$selected_mount_point_docker_root_dir" && \
        sudo rsync -axPS /var/lib/docker/ "$selected_mount_point_docker_root_dir" && \
        sudo du -csh "$selected_mount_point_docker_root_dir"
    
    # Update Docker daemon.json with new data-root
    sudo jq --arg dataroot "$selected_mount_point_docker_root_dir" \
        '. + {"data-root": $dataroot}' /etc/docker/daemon.json | \
        sudo tee /etc/docker/daemon.json.tmp > /dev/null
    
    # Validate JSON before replacing daemon.json
    if sudo jq empty /etc/docker/daemon.json.tmp >/dev/null 2>&1; then
        sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
        echo "✅ Docker data-root updated to: $selected_mount_point_docker_root_dir"
    else
        echo "❌ JSON validation failed. Aborting update."
        rm /etc/docker/daemon.json.tmp
        return 1
    fi
    
    # Rename the old Docker data directory
    sudo mv /var/lib/docker /var/lib/docker.old
    
    # Restart the docker daemon
    sudo systemctl daemon-reload && \
        sudo systemctl restart docker
}

################################################################################
# HELP AND ARGUMENT PARSING
################################################################################

print_help() {
    echo -e "\n\033[1;34mJetson Docker Configuration Script\033[0m"
    echo -e "\n\033[1;34mUsage:\033[0m $0 [OPTIONS]"
    echo
    echo -e "\033[1;36mOptions:\033[0m"
    echo -e "  --help             Show this help message and exit."
    echo -e "  --test=<function>  Run a specific function for testing."
    echo -e "  --no-migrate      Skip Docker data directory migration."
    echo
    echo -e "\033[1;36mDescription:\033[0m"
    echo "  This script configures Docker on Jetson devices with proper NVIDIA"
    echo "  container runtime support and optionally migrates Docker data to NVMe storage."
    echo
    echo -e "\033[1;36mSteps Performed:\033[0m"
    echo "  1️⃣ Install Docker if it's not already installed."
    echo "  2️⃣ Configure default container runtime to NVIDIA."
    echo "  3️⃣ If using eMMC + NVMe, migrate Docker data-root to SSD."
    echo "  4️⃣ Ensure current user is in the docker group."
    echo
}

parse_args() {
    TEST_FUNCTION=""
    NO_MIGRATE=false
    
    for arg in "$@"; do
        case "$arg" in
            --test=*)
                TEST_FUNCTION="${arg#*=}"
                ;;
            --no-migrate)
                NO_MIGRATE=true
                ;;
            --help)
                print_help
                exit 0
                ;;
            *)
                log ERROR "Unknown parameter: $arg"
                exit 1
                ;;
        esac
    done
}

################################################################################
# MAIN EXECUTION FLOW
################################################################################

main() {
    parse_args "$@"
    
    # Handle test function execution
    if [[ -n "$TEST_FUNCTION" ]]; then
        if declare -F "$TEST_FUNCTION" > /dev/null; then
            log WARN "Running test for function: $TEST_FUNCTION"
            "$TEST_FUNCTION"
            exit 0
        else
            log ERROR "Function '$TEST_FUNCTION' not found."
            exit 1
        fi
    fi
    
    # Normal execution flow
    print_section "Installing and configuring Docker"
    if [[ "$NO_MIGRATE" != true ]]; then
        setup_docker
    else
        configure_docker_runtime
        setup_docker_group
    fi
    
    print_section "=== SETUP COMPLETED ==="
    log INFO "Docker is now configured with NVIDIA runtime as default"
    
    # Remind user to log out and back in if group was added
    if ! groups $USER | grep -q '\bdocker\b'; then
        log WARN "⚠️ IMPORTANT: Log out and back in for docker group changes to take effect"
        log WARN "              Alternatively, run 'newgrp docker' in your current terminal"
    fi
}

# Execute main function
main "$@"