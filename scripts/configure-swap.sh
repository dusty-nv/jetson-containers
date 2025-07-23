#!/bin/bash

# Enable error handling
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load environment variables from .env
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
elif [ -f "${SCRIPT_DIR}/../.env" ]; then
    set -a
    source "${SCRIPT_DIR}/../.env"
    set +a
fi

# Default values
interactive_mode="${INTERACTIVE_MODE:-true}"
# Use SWAP_OPTIONS_SIZE from the .env file if present, otherwise default to 8G
swap_size_num="${SWAP_OPTIONS_SIZE:-8}"
swap_size="${swap_size_num}G"
# Use SWAP_OPTIONS_DISABLE_ZRAM to determine if zRAM should be disabled by default
zram_disabled="${SWAP_OPTIONS_DISABLE_ZRAM:-false}"
zram_enabled="ask"
if [ "$zram_disabled" = "true" ]; then
    zram_enabled="no"
fi
zram_size="${ZRAM_SIZE:-4G}"
# Use SWAP_OPTIONS_PATH if specified, otherwise default to /swapfile
swap_file_path="${SWAP_OPTIONS_PATH:-/swapfile}"

# Function to prompt yes/no questions
ask_yes_no() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or y or no or n.";;
        esac
    done
}

# Convert size string to bytes (e.g., "4G" to bytes)
convert_to_bytes() {
    local size=$1
    local value=${size%[GM]}
    local unit=${size#$value}
    
    case $unit in
        G)
            echo $((value * 1024 * 1024 * 1024))
            ;;
        M)
            echo $((value * 1024 * 1024))
            ;;
        *)
            echo "Error: Invalid size unit. Use M for MB or G for GB"
            exit 1
            ;;
    esac
}

# Check if swap file exists and is active
check_swap_file() {
    if [ -f "$swap_file_path" ]; then
        if swapon -s | grep -q "$swap_file_path"; then
            echo "active"
        else
            echo "inactive"
        fi
    else
        echo "missing"
    fi
}

# Create and enable swap file
setup_swap_file() {
    local swap_state=$(check_swap_file)
    local size_bytes=$(convert_to_bytes "$swap_size")
    
    # If swap file exists, disable and remove it first
    if [ "$swap_state" != "missing" ]; then
        echo "Disabling existing swap file..."
        sudo swapoff "$swap_file_path" || true
        echo "Removing existing swap file..."
        sudo rm -f "$swap_file_path"
        # Remove from fstab if present
        sudo sed -i "\\#^${swap_file_path}#d" /etc/fstab
    fi
    
    # Create new swap file directly at the final location
    echo "Creating new swap file of size $swap_size..."
    sudo dd if=/dev/zero of="$swap_file_path" bs=1M count=$((size_bytes/1024/1024))
    sudo chmod 600 "$swap_file_path"
    sudo mkswap "$swap_file_path"
    sudo swapon "$swap_file_path"
    
    # Add to fstab if not already present
    if ! grep -q "$swap_file_path" /etc/fstab; then
        echo "$swap_file_path none swap sw 0 0" | sudo tee -a /etc/fstab
    fi
    
    if swapon -s | grep -q "$swap_file_path"; then
        echo "✅ Swap file setup complete"
        return 0
    else
        echo "❌ Failed to setup swap file"
        return 1
    fi
}

# Check if zRAM is enabled
check_zram() {
    if lsmod | grep -q "^zram"; then
        echo "active"
    elif [ -f "/etc/modules-load.d/zram.conf" ]; then
        echo "configured"
    else
        echo "disabled"
    fi
}

# Setup zRAM
setup_zram() {
    local zram_state=$(check_zram)
    local size_bytes=$(convert_to_bytes "$zram_size")
    
    # If zRAM should be enabled
    if [ "$zram_enabled" = "yes" ]; then
        # Create configuration files if they don't exist
        echo "zram" | sudo tee /etc/modules-load.d/zram.conf
        echo "options zram num_devices=1" | sudo tee /etc/modprobe.d/zram.conf
        
        # Create udev rule for zRAM configuration
        cat << EOF | sudo tee /etc/udev/rules.d/99-zram.rules
KERNEL=="zram0", ATTR{disksize}="$size_bytes", TAG+="systemd"
EOF
        
        # Create systemd service for zRAM
        cat << EOF | sudo tee /etc/systemd/system/zram.service
[Unit]
Description=zRAM setup
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/sbin/mkswap /dev/zram0
ExecStart=/sbin/swapon -p 5 /dev/zram0
ExecStop=/sbin/swapoff /dev/zram0
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
        
        # Enable and start zRAM
        sudo modprobe zram
        sudo systemctl daemon-reload
        sudo systemctl enable zram
        sudo systemctl start zram
        
        if [ "$(check_zram)" = "active" ]; then
            echo "✅ zRAM setup complete"
            return 0
        else
            echo "❌ Failed to setup zRAM"
            return 1
        fi
    else
        # Disable zRAM if it's enabled
        if [ "$zram_state" != "disabled" ]; then
            sudo systemctl stop zram
            sudo systemctl disable zram
            sudo swapoff /dev/zram0 2>/dev/null || true
            sudo rmmod zram 2>/dev/null || true
            sudo rm -f /etc/modules-load.d/zram.conf /etc/modprobe.d/zram.conf \
                      /etc/udev/rules.d/99-zram.rules /etc/systemd/system/zram.service
            echo "✅ zRAM has been disabled"
        else
            echo "zRAM is already disabled"
        fi
    fi
}

# Configure memory settings
configure_memory() {
    # Setup swap file
    echo "=== Configuring Swap File ==="
    setup_swap_file
    
    # Configure zRAM if needed
    echo -e "\n=== Configuring zRAM ==="
    if [ "$zram_enabled" = "ask" ] && [ "$interactive_mode" = "true" ]; then
        if ask_yes_no "Would you like to disable zRAM?"; then
            zram_enabled="no"
        else
            zram_enabled="yes"
        fi
    fi
    setup_zram
    
    # Display current memory configuration
    echo -e "\n=== Current Memory Configuration ==="
    free -h
    echo -e "\nSwap configuration:"
    swapon -s
}

# Parse command line arguments
parse_args() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --swap-size=*)
                swap_size="${1#*=}"
                shift
                ;;
            --zram-size=*)
                zram_size="${1#*=}"
                shift
                ;;
            --enable-zram)
                zram_enabled="yes"
                shift
                ;;
            --disable-zram)
                zram_enabled="no"
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Configure swap file and zRAM"
                echo ""
                echo "Options:"
                echo "  --swap-size=SIZE  Set swap file size (e.g., 8G, 4096M)"
                echo "  --zram-size=SIZE  Set zRAM size (e.g., 4G, 2048M)"
                echo "  --enable-zram     Enable zRAM"
                echo "  --disable-zram    Disable zRAM"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown parameter passed: $1"
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    if [ "$EUID" -ne 0 ]; then 
        echo "Please run as root (with sudo)"
        exit 1
    fi
    
    parse_args "$@"
    configure_memory
}

# Run main function with arguments
main "$@"