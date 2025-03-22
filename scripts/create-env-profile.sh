#!/bin/bash
# Script to generate .env configuration files for different Jetson device profiles

#set -e

# Default configuration directory
CONFIG_DIR="$(dirname "$(realpath "$0")")"
ENV_FILE="${CONFIG_DIR}/../.env"
# Function to detect current device
detect_device() {
    if [ -f /etc/nv_tegra_release ]; then
        if grep -q "AGX Orin" /proc/device-tree/model 2>/dev/null; then
            echo "agx"
        elif grep -q "Orin NX" /proc/device-tree/model 2>/dev/null; then
            echo "nx"
        elif grep -q "Orin Nano" /proc/device-tree/model 2>/dev/null; then
            echo "nano"
        else
            echo "unknown"
        fi
    else
        echo "not-jetson"
    fi
}

# Function to detect storage configuration
detect_storage_config() {
    local device=$1
    local boot_device
    
    # For debugging - print mount info but don't include in output
    echo "DEBUG: Mount information for root filesystem:" >&2
    mount | grep " on / " | tee /dev/stderr >/dev/null
    
    # Determine boot device
    if mount | grep -q " on / .*/dev/nvme" || lsblk | grep -q "nvme.*/$"; then
        boot_device="nvme"
        echo "DEBUG: Detected NVMe boot device" >&2
    elif mount | grep -q " on / .*/dev/mmcblk0" || mount | grep -q " on / .*mmcblk0"; then
        if [[ "$device" == "agx" ]]; then
            boot_device="emmc"
            echo "DEBUG: Detected eMMC boot device on AGX" >&2
        else
            boot_device="sdcard"
            echo "DEBUG: Detected SD card boot device" >&2
        fi
    else
        # If we can't determine the boot device, check if it's likely eMMC on AGX
        if [[ "$device" == "agx" ]] && lsblk | grep -q "mmcblk0"; then
            boot_device="emmc"
            echo "DEBUG: Fallback detection found likely eMMC boot device" >&2
        elif [[ "$device" == "nano" ]] && lsblk | grep -q "mmcblk0"; then
            boot_device="sdcard"
            echo "DEBUG: Fallback detection found likely SD card boot device" >&2
        else
            boot_device="unknown"
            echo "DEBUG: Could not determine boot device type" >&2
        fi
    fi
    
    # Check for additional NVMe
    local has_nvme=false
    if lsblk | grep -q "nvme"; then
        has_nvme=true
        echo "DEBUG: NVMe storage detected" >&2
    fi
    
    # Return combined device and storage info
    echo "${device}-${boot_device}${has_nvme:+-nvme}"
}

# Function to generate profile configuration
generate_profile_config() {
    local profile=$1
    
    # Common configuration
    local config="# Environment configuration for ${profile}\n"
    config+="INTERACTIVE_MODE=true\n\n"

    # Device-specific settings
    if [[ "$profile" == "agx"* ]]; then
        config+="# Power mode (0=MAXN, 1=15W, 2=30W, 3=50W)\n"
        config+="POWER_MODE_SHOULD_RUN=yes\n"
        config+="POWER_MODE_OPTIONS_MODE=2\n\n"
    elif [[ "$profile" == "nano"* ]]; then
        config+="# Power mode (0=MAXN, 1=MODE_15W, etc.)\n"
        config+="POWER_MODE_SHOULD_RUN=yes\n"
        config+="POWER_MODE_OPTIONS_MODE=2\n\n"
    fi

    # Add NVMe and Docker configuration
    if [[ "$profile" == *"-nvme" ]]; then
        config+="# NVMe configuration\n"
        config+="NVME_SETUP_SHOULD_RUN=yes\n"
        config+="NVME_SETUP_OPTIONS_MOUNT_POINT=/mnt\n"
        config+="NVME_SETUP_OPTIONS_PARTITION_NAME=nvme0n1p1\n"
        config+="NVME_SETUP_OPTIONS_FILESYSTEM=ext4\n\n"
        
        config+="# Docker configuration\n"
        config+="DOCKER_RUNTIME_SHOULD_RUN=yes\n"
        config+="DOCKER_ROOT_SHOULD_RUN=yes\n"
        config+="DOCKER_ROOT_OPTIONS_PATH=/mnt/docker\n\n"
    else
        config+="# Docker configuration\n"
        config+="DOCKER_RUNTIME_SHOULD_RUN=yes\n"
        config+="DOCKER_ROOT_SHOULD_RUN=no\n\n"
    fi
    
    # Swap configuration
    config+="# Swap configuration\n"
    config+="SWAP_SHOULD_RUN=yes\n"
    config+="SWAP_OPTIONS_DISABLE_ZRAM=true\n"
    
    # Adjust swap size based on the device
    if [[ "$profile" == "agx"* ]]; then
        config+="SWAP_OPTIONS_SIZE=32\n"
        if [[ "$profile" == *"-nvme" ]]; then
            config+="SWAP_OPTIONS_PATH=/mnt/32GB.swap\n\n"
        else
            config+="SWAP_OPTIONS_PATH=/swapfile\n\n"
        fi
    elif [[ "$profile" == "nano"* ]]; then
        config+="SWAP_OPTIONS_SIZE=8\n"
        if [[ "$profile" == *"-nvme" ]]; then
            config+="SWAP_OPTIONS_PATH=/mnt/8GB.swap\n\n"
        else
            config+="SWAP_OPTIONS_PATH=/swapfile\n\n"
        fi
    fi
    
    # GUI settings
    config+="# GUI configuration\n"
    config+="GUI_DISABLED_SHOULD_RUN=ask\n\n"

    # Docker group settings
    config+="# Docker group configuration\n"
    config+="DOCKER_GROUP_SHOULD_RUN=yes\n"
    config+="DOCKER_GROUP_OPTIONS_ADD_USER=$(whoami)\n\n"
    
    echo -e "$config"
}

# Simple manual profile selection function
create_manual_profile() {
    local DEVICE=$1
    
    # Get OS storage type
    echo "What is your OS storage type?"
    echo "1) eMMC (for AGX Orin)"
    echo "2) SD card (for Orin Nano)"
    echo "3) NVMe"
    
    local os_storage
    read -p "Enter selection (1-3): " os_choice
    case $os_choice in
        1) os_storage="emmc" ;;
        2) os_storage="sdcard" ;;
        3) os_storage="nvme" ;;
        *) echo "Invalid choice. Using 'unknown'."; os_storage="unknown" ;;
    esac
    
    # Check for additional NVMe
    echo "Do you have additional NVMe storage?"
    echo "1) Yes"
    echo "2) No"
    
    local has_nvme=false
    read -p "Enter selection (1-2): " nvme_choice
    case $nvme_choice in
        1) has_nvme=true ;;
        2) has_nvme=false ;;
        *) echo "Invalid choice. Assuming no additional NVMe."; has_nvme=false ;;
    esac
    
    # Generate profile name
    local profile="${DEVICE}-${os_storage}"
    if $has_nvme; then
        profile="${profile}-nvme"
    fi
    
    echo "Generated profile: $profile"
    echo "$profile"
}

# Function to automatically select appropriate profile based on detected hardware
auto_select_profile() {
    local detected_profile=$1
    
    echo "DEBUG: Auto-selecting profile based on: $detected_profile" >&2
    
    if [[ "$detected_profile" == "agx-nvme"* ]]; then
        echo "agx-nvme"
    elif [[ "$detected_profile" == "agx-emmc-nvme" ]]; then
        echo "agx-emmc-nvme"
    elif [[ "$detected_profile" == "agx-emmc" ]]; then
        echo "agx-emmc"
    elif [[ "$detected_profile" == "nano-nvme"* ]]; then
        echo "nano-nvme"
    elif [[ "$detected_profile" == "nano-sdcard-nvme" ]]; then
        echo "nano-sdcard-nvme"
    elif [[ "$detected_profile" == "nano-sdcard" ]]; then
        echo "nano-sdcard"
    elif [[ "$detected_profile" == "agx-unknown-nvme" ]]; then
        # Special handling for unknown boot device with NVMe
        echo "agx-emmc-nvme"
    elif [[ "$detected_profile" == "nano-unknown-nvme" ]]; then
        # Special handling for unknown boot device with NVMe
        echo "nano-sdcard-nvme"
    else
        # Use a reasonable default based on the detected device
        if [[ "$detected_profile" == "agx"* ]]; then
            echo "agx-emmc"
        elif [[ "$detected_profile" == "nano"* ]]; then
            echo "nano-sdcard"
        else
            echo "agx-emmc" # Default fallback
        fi
    fi
}

# Function to save configuration to file
save_config() {
    local config=$1
    local file=$2
    
    # Create backup if file exists
    if [[ -f "$file" ]]; then
        backup="${file}.$(date +%Y%m%d%H%M%S).bak"
        echo "Creating backup of existing .env file: $backup"
        cp "$file" "$backup"
    fi
    
    echo -e "$config" > "$file"
    echo "Configuration saved to $file"
}

# Parse command line arguments
DEVICE=""
INTERACTIVE=false
PROFILE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --profile-override)
            PROFILE_OVERRIDE="$2"
            shift 2
            ;;
        --profile-override=*)
            PROFILE_OVERRIDE="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --device <device>    Specify device (agx, nano)"
            echo "  --interactive        Run in interactive mode (with prompts)"
            echo "  --profile-override <profile>  Override detected profile with specified profile"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect device if not specified
if [[ -z "$DEVICE" ]]; then
    DEVICE=$(detect_device)
fi

# Check for supported devices
if [[ "$DEVICE" == "nx" ]]; then
    echo "Jetson Orin NX is currently not supported by this script."
    echo "Please configure your .env file manually using the examples in the documentation."
    exit 1
elif [[ "$DEVICE" == "not-jetson" ]]; then
    echo "This doesn't appear to be a Jetson device."
    echo "Please specify a device type with --device if you're sure this is a Jetson."
    exit 1
elif [[ "$DEVICE" == "unknown" ]]; then
    echo "Unknown Jetson device type."
    if [[ "$INTERACTIVE" == "true" ]]; then
        echo "Please specify your device type:"
        echo
        echo "1) Jetson AGX Orin"
        echo "2) Jetson Orin Nano"
        
        read -p "Enter selection (1-2): " device_selection
        case $device_selection in
            1) DEVICE="agx" ;;
            2) DEVICE="nano" ;;
            *) 
                echo "Invalid selection. Defaulting to AGX Orin."
                DEVICE="agx"
            ;;
        esac
    else
        echo "Defaulting to AGX Orin."
        DEVICE="agx"
    fi
fi

# Detect storage configuration
DETECTED_CONFIG=$(detect_storage_config "$DEVICE")
echo "Detected device configuration: $DETECTED_CONFIG"

# Force a known configuration for now to avoid hanging
if [[ "$DETECTED_CONFIG" == *"unknown"* ]]; then
    # Instead of forcing a default configuration, let auto_select_profile handle it
    echo "Unable to precisely determine boot device configuration."
    echo "Will select appropriate profile based on available information."
fi

# Auto-select profile without user interaction unless interactive mode is specified
if [[ -n "$PROFILE_OVERRIDE" ]]; then
    SELECTED_PROFILE="$PROFILE_OVERRIDE"
    echo "Profile override specified: $SELECTED_PROFILE"
elif [[ "$INTERACTIVE" == "true" ]]; then
    echo "Running in interactive mode..."
    echo
    echo "======================================================="
    echo "          JETSON ENVIRONMENT PROFILE GENERATOR         "
    echo "======================================================="
    echo
    if [[ "$DETECTED_CONFIG" == "agx"* ]]; then
        echo "DETECTED DEVICE: JETSON AGX ORIN"
        
        echo
        echo "Please select a profile for your AGX Orin:"
        echo
        echo "  1) AGX Orin with OS on eMMC"
        echo "  2) AGX Orin with OS on eMMC + NVMe storage"
        echo "  3) AGX Orin with OS on NVMe"
        echo "  4) AGX Orin with OS on NVMe + additional NVMe"
        echo "  5) Manual configuration"
        echo
        
        local selection
        read -p "Enter selection (1-5): " selection
        
        case $selection in
            1) SELECTED_PROFILE="agx-emmc" ;;
            2) SELECTED_PROFILE="agx-emmc-nvme" ;;
            3) SELECTED_PROFILE="agx-nvme" ;;
            4) SELECTED_PROFILE="agx-nvme-nvme" ;;
            5) SELECTED_PROFILE=$(create_manual_profile "agx") ;;
            *) 
                echo "Invalid selection. Using detected configuration."
                SELECTED_PROFILE="$DETECTED_CONFIG" 
            ;;
        esac
    elif [[ "$DETECTED_CONFIG" == "nano"* ]]; then
        echo "DETECTED DEVICE: JETSON ORIN NANO"
        
        echo
        echo "Please select a profile for your Orin Nano:"
        echo
        echo "  1) Orin Nano with OS on SD card"
        echo "  2) Orin Nano with OS on SD card + NVMe storage"
        echo "  3) Orin Nano with OS on NVMe"
        echo "  4) Orin Nano with OS on NVMe + additional NVMe"
        echo "  5) Manual configuration"
        echo
        
        local selection
        read -p "Enter selection (1-5): " selection
        
        case $selection in
            1) SELECTED_PROFILE="nano-sdcard" ;;
            2) SELECTED_PROFILE="nano-sdcard-nvme" ;;
            3) SELECTED_PROFILE="nano-nvme" ;;
            4) SELECTED_PROFILE="nano-nvme-nvme" ;;
            5) SELECTED_PROFILE=$(create_manual_profile "nano") ;;
            *) 
                echo "Invalid selection. Using detected configuration."
                SELECTED_PROFILE="$DETECTED_CONFIG" 
            ;;
        esac
    else
        echo "UNKNOWN OR CUSTOM DEVICE"
        
        echo
        echo "Please select a device type:"
        echo
        echo "  1) Jetson AGX Orin"
        echo "  2) Jetson Orin Nano"
        echo
        
        local device
        read -p "Enter selection (1-2): " device_selection
        
        case $device_selection in
            1) device="agx" ;;
            2) device="nano" ;;
            *) 
                echo "Invalid selection. Defaulting to AGX Orin."
                device="agx" 
            ;;
        esac
        
        SELECTED_PROFILE=$(create_manual_profile "$device")
    fi
else
    # Auto-select profile based on detected hardware
    SELECTED_PROFILE=$(auto_select_profile "$DETECTED_CONFIG")
    echo "Automatically selected profile: $SELECTED_PROFILE"
fi

# Generate and save profile configuration
CONFIG=$(generate_profile_config "$SELECTED_PROFILE")
save_config "$CONFIG" "$ENV_FILE"

echo
echo "======================================================="
echo "    Environment configuration created successfully!    "
echo "======================================================="
echo

# Run initial system probe
echo "Current system status:"
echo "---------------------"
"${CONFIG_DIR}/probe-system.sh"
echo

echo "Next steps:"
echo "  1. Review the configuration in ${ENV_FILE}"
echo "  2. Run: sudo ./scripts/setup-system.sh"
echo "     This will automatically:"
if [[ "$SELECTED_PROFILE" == *"-nvme" ]]; then
    echo "     - Set up NVMe storage at ${NVME_SETUP_OPTIONS_MOUNT_POINT}"
    echo "     - Configure Docker to use ${DOCKER_ROOT_OPTIONS_PATH}"
fi
echo "     - Configure system swap (${SWAP_OPTIONS_SIZE}GB)"
echo "     - Set up power mode and other system settings"
echo "  3. Reboot to apply all changes"
echo