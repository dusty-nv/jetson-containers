#!/bin/bash

# State file location
STATE_FILE="initialize-jetson-state.yaml"

# Check if script is run with sudo
check_permissions() {
    if [ "$EUID" -ne 0 ]; then 
        echo "Please run as root (with sudo)"
        exit 1
    fi
}
# Helper function to get indentation level of a line
get_indent_level() {
    local line="$1"
    local spaces=$(echo "$line" | sed 's/^\([[:space:]]*\).*/\1/')
    echo $((${#spaces} / 2))
}

# Helper function to get the parent path of a key
get_parent_path() {
    local key="$1"
    echo "${key%.*}"
}

yaml_set() {
    local yaml_file=$1
    local key=$2
    local value=$3
    local temp_file="${yaml_file}.tmp"
    
    # Create temp file if it doesn't exist
    touch "$temp_file"
    
    # Split the key into parts
    IFS='.' read -ra key_parts <<< "$key"
    
    # Initialize variables
    local current_indent=0
    local path_stack=()
    local in_target_path=false
    local found_key=false
    local last_line_processed=false
    
    # Process the input file line by line
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines, but preserve them
        if [[ $line =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            echo "$line" >> "$temp_file"
            continue
        fi
        
        # Get indentation level
        local indent_level=$(get_indent_level "$line")
        
        # Extract the current line's key
        if [[ $line =~ ^[[:space:]]*([^:]+): ]]; then
            local current_key="${BASH_REMATCH[1]}"
            
            # Update path stack based on indentation
            while [ ${#path_stack[@]} -gt $indent_level ]; do
                unset 'path_stack[${#path_stack[@]}-1]'
            done
            
            if [ $indent_level -eq 0 ]; then
                path_stack=("$current_key")
            else
                path_stack[$indent_level]="$current_key"
            fi
            
            # Build current path
            local current_path=""
            for ((i=0; i<${#path_stack[@]}; i++)); do
                [ -n "$current_path" ] && current_path="${current_path}."
                current_path="${current_path}${path_stack[i]}"
            done
            
            # Check if we're at our target key
            if [ "$current_path" = "$key" ]; then
                echo "${line%:*}: $value" >> "$temp_file"
                found_key=true
                continue
            fi
        fi
        
        # Write the current line to the temp file
        echo "$line" >> "$temp_file"
    done < "$yaml_file"
    
    # If key wasn't found, add it with proper indentation
    if [ "$found_key" = false ]; then
        local parent_path=$(get_parent_path "$key")
        local final_key="${key##*.}"
        local indent=""
        
        # Calculate proper indentation based on parent path depth
        if [ -n "$parent_path" ]; then
            local depth=$(echo "$parent_path" | grep -o "\." | wc -l)
            depth=$((depth + 1))
            for ((i=0; i<depth; i++)); do
                indent="${indent}  "
            done
        fi
        
        # Add the new key-value pair
        echo "${indent}${final_key}: ${value}" >> "$temp_file"
    fi
    
    # Replace original file with temp file
    mv "$temp_file" "$yaml_file"
}

yaml_get() {
    local yaml_file=$1
    local search_key=$2
    local value=""
    local current_path=""
    local current_indent=0
    local path_stack=()
    
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ $line =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        
        # Calculate indentation level
        local spaces=$(echo "$line" | sed 's/^\([[:space:]]*\).*/\1/')
        local indent=$((${#spaces} / 2))
        
        # Extract key and value
        if [[ $line =~ ^[[:space:]]*([^:]+):[[:space:]]*(.*) ]]; then
            local key="${BASH_REMATCH[1]}"
            local val="${BASH_REMATCH[2]}"
            
            # Handle indentation changes
            if [ $indent -lt $current_indent ]; then
                for ((i=0; i<current_indent-indent; i++)); do
                    unset 'path_stack[${#path_stack[@]}-1]'
                done
            fi
            
            # Update path stack
            if [ $indent -eq 0 ]; then
                path_stack=("$key")
            else
                if [ $indent -gt $current_indent ]; then
                    path_stack+=("$key")
                else
                    path_stack[$indent]="$key"
                fi
            fi
            
            # Build current path
            current_path=""
            for ((i=0; i<${#path_stack[@]}; i++)); do
                [ -n "$current_path" ] && current_path="${current_path}."
                current_path="${current_path}${path_stack[i]}"
            done
            
            # Check if this is our target
            if [ "$current_path" = "$search_key" ]; then
                value="${val## }"
                break
            fi
            
            current_indent=$indent
        fi
    done < "$yaml_file"
    
    echo "$value"
}

# Helper function to test if a key exists in YAML
yaml_key_exists() {
    local yaml_file=$1
    local key=$2
    local value=$(yaml_get "$yaml_file" "$key")
    [ ! -z "$value" ]
    return $?
}

# Check for minimal dependencies
check_dependencies() {
    local missing_deps=()
    
    # Required commands
    if ! command -v nvpmodel &> /dev/null; then
        missing_deps+=("nvpmodel")
    fi
    if ! command -v parted &> /dev/null; then
        missing_deps+=("parted")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Error: Missing required dependencies: ${missing_deps[*]}"
        echo "Please install them using:"
        echo "sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
}

# Initialize or load state
init_state() {
    ORIGINAL_USER=$(logname)
    if [ ! -f "$STATE_FILE" ]; then
        # Create default YAML configuration
        cat > "$STATE_FILE" <<EOL
# System Configuration
interactive_mode: true

system:
  nvme_status: unknown

# NVMe Setup
nvme_setup:
  done: false
  should_run: ask
  options:
    mount_point: /mnt
    partition_name: nvme0n1p1
    filesystem: ext4

# Docker Configuration
docker_runtime:
  done: false
  should_run: ask
  options: {}

docker_root:
  done: false
  should_run: ask
  path: ""
  options:
    path: /mnt/docker

# Memory Configuration
swap:
  done: false
  should_run: ask
  size: ""
  path: ""
  options:
    disable_zram: true
    size: "16"
    path: /mnt/16GB.swap

# System Settings
gui_disabled:
  done: false
  should_run: ask
  options: {}

docker_group:
  done: false
  should_run: ask
  options: {}

power_mode:
  done: false
  should_run: ask
  options:
    mode: "1"  # 25W mode (recommended)
EOL
    fi
}

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

# Function to check if action should run
should_run() {
    local action=$1
    local status=$(yaml_get "$STATE_FILE" "${action}.should_run")
    
    case $status in
        "yes") return 0;;
        "no") return 1;;
        "ask"|*) 
            local interactive=$(yaml_get "$STATE_FILE" "interactive_mode")
            if [ "$interactive" = "true" ]; then
                ask_yes_no "$2"
                return $?
            else
                return 1
            fi
            ;;
    esac
}

# Function to detect NVMe and OS location
detect_nvme_setup() {
    if [ -b "/dev/nvme0n1" ]; then
        if mount | grep "/dev/nvme0n1" | grep -q " / "; then
            yaml_set "$STATE_FILE" "system.nvme_status" "os_on_nvme"
            echo "Detected OS running on NVMe drive"
        else
            # Check for existing mounts
            if mount | grep -q "/dev/nvme0n1"; then
                local current_mount=$(mount | grep "/dev/nvme0n1" | head -n1 | awk '{ print $3 }')
                yaml_set "$STATE_FILE" "system.nvme_status" "nvme_mounted"
                yaml_set "$STATE_FILE" "system.current_mount" "$current_mount"
                echo "Note: NVMe drive is mounted at: $current_mount"
            else
                yaml_set "$STATE_FILE" "system.nvme_status" "nvme_available"
                echo "Detected unused NVMe drive"
            fi
        fi
    else
        yaml_set "$STATE_FILE" "system.nvme_status" "no_nvme"
        echo "No NVMe drive detected"
    fi
}

# Setup NVMe if available
setup_nvme() {
    if [ "$(yaml_get "$STATE_FILE" "nvme_setup.done")" = "true" ]; then
        echo "NVMe already configured, skipping..."
        return 0
    fi

    local nvme_status=$(yaml_get "$STATE_FILE" "system.nvme_status")
    local desired_mount=$(yaml_get "$STATE_FILE" "nvme_setup.options.mount_point")

    if [ "$nvme_status" = "no_nvme" ]; then
        echo "No NVMe drive detected, skipping NVMe setup..."
        return 0
    elif [ "$nvme_status" = "os_on_nvme" ]; then
        echo "OS is running on NVMe, skipping additional NVMe setup..."
        return 0
    elif [ "$nvme_status" = "nvme_mounted" ]; then
        local current_mount=$(yaml_get "$STATE_FILE" "system.current_mount")
        echo "NVMe is mounted at $current_mount"
        
        if [ "$desired_mount" != "$current_mount" ]; then
            if should_run "nvme_setup" "Would you like to create a symlink from $desired_mount to the existing NVMe mount?"; then
                # Remove existing mount point if it's empty
                if [ -d "$desired_mount" ] && [ -z "$(ls -A "$desired_mount")" ]; then
                    rmdir "$desired_mount"
                fi
                
                # Create symlink if it doesn't exist
                if [ ! -e "$desired_mount" ]; then
                    ln -s "$current_mount" "$desired_mount"
                    echo "Created symlink: $desired_mount -> $current_mount"
                    yaml_set "$STATE_FILE" "nvme_setup.done" "true"
                    return 0
                else
                    echo "Error: $desired_mount already exists and is not empty"
                    return 1
                fi
            fi
        fi
        return 0
    fi

    # Full setup for unused NVMe drive
    if should_run "nvme_setup" "Would you like to setup the NVMe drive?"; then
        local mount_point="$desired_mount"
        local partition_name=$(yaml_get "$STATE_FILE" "nvme_setup.options.partition_name")
        local filesystem=$(yaml_get "$STATE_FILE" "nvme_setup.options.filesystem")

        echo "Setting up NVMe drive..."
        
        # Check partition and get user confirmation for any changes
        if [ ! -b "/dev/$partition_name" ]; then
            if ask_yes_no "No partition found (/dev/$partition_name). Would you like to create a new partition on the NVMe drive? (WARNING: This will erase all data)"; then
                echo "Creating partition on NVMe drive..."
                parted /dev/nvme0n1 mklabel gpt
                parted /dev/nvme0n1 mkpart primary ext4 0% 100%
                sleep 2  # Wait for partition to be recognized
            else
                echo "Skipping partition creation"
                return 0
            fi
        fi

        # Also confirm formatting if needed
        if [ -b "/dev/$partition_name" ] && ! blkid "/dev/$partition_name" | grep -q "$filesystem"; then
            if ask_yes_no "Partition needs formatting. Would you like to format as $filesystem? (WARNING: This will erase all data)"; then
                echo "Formatting NVMe partition as $filesystem..."
                mkfs.$filesystem "/dev/$partition_name"
            else
                echo "Skipping formatting"
                return 0
            fi
        fi

        # Create mount point if it doesn't exist
        if [ ! -d "$mount_point" ]; then
            mkdir -p "$mount_point"
        fi

        # Add to fstab if not already present
        if ! grep -q "/dev/$partition_name" /etc/fstab; then
            echo "Adding NVMe mount to fstab..."
            echo "/dev/$partition_name $mount_point $filesystem defaults 0 0" >> /etc/fstab
        fi

        # Mount the drive
        if ! mount | grep -q "/dev/$partition_name"; then
            mount "/dev/$partition_name"
        fi

        yaml_set "$STATE_FILE" "nvme_setup.done" "true"
        return 0
    fi
    return 0
}

# Configure Docker runtime
setup_docker_runtime() {
    if [ "$(yaml_get "$STATE_FILE" "docker_runtime.done")" = "true" ]; then
        echo "Docker runtime already configured, skipping..."
        return 0
    fi

    if should_run "docker_runtime" "Would you like to configure Docker runtime?"; then
        echo "Configuring Docker runtime..."
        cat > /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF
        yaml_set "$STATE_FILE" "docker_runtime.done" "true"
        return 0
    fi
    return 0
}

# Configure Docker data root
setup_docker_root() {
    if [ "$(yaml_get "$STATE_FILE" "docker_root.done")" = "true" ]; then
        echo "Docker root already configured, skipping..."
        return 0
    fi

    if should_run "docker_root" "Would you like to relocate Docker data root?"; then
        local docker_root=$(yaml_get "$STATE_FILE" "docker_root.options.path")
        if [ -z "$docker_root" ]; then
            echo "No Docker root path found in options, using default path"
            docker_root="/mnt/docker"
        fi
        echo "Using Docker root path: $docker_root"
        
        if [ ! -d "$docker_root" ]; then
            echo "Creating directory: $docker_root"
            mkdir -p "$docker_root"
        fi

        if [ -d "$docker_root" ]; then
            echo "Copying existing Docker data..."
            if cp -r /var/lib/docker/* "$docker_root/"; then
                # Update daemon.json
                sed -i 's/}\s*$/,\n    "data-root": "'"$docker_root"'"\n}/' /etc/docker/daemon.json
                
                yaml_set "$STATE_FILE" "docker_root.done" "true"
                yaml_set "$STATE_FILE" "docker_root.path" "$docker_root"
                return 0
            else
                echo "Failed to copy Docker data"
                return 1
            fi
        else
            echo "Failed to create directory $docker_root"
            return 1
        fi
    fi
    return 0
}

setup_swap() {
    if [ "$(yaml_get "$STATE_FILE" "swap.done")" = "true" ]; then
        echo "Swap already configured, skipping..."
        return 0
    fi

    if should_run "swap" "Would you like to configure swap space?"; then
        local disable_zram=$(yaml_get "$STATE_FILE" "swap.options.disable_zram")
        local swap_size=$(yaml_get "$STATE_FILE" "swap.options.size" | tr -d '"')  # Remove quotes
        local swap_file=$(yaml_get "$STATE_FILE" "swap.options.path" | tr -d '"')  # Remove quotes

        # Validate inputs
        if [ -z "$swap_size" ]; then
            echo "Error: No swap size specified in configuration"
            return 1
        fi
        if [ -z "$swap_file" ]; then
            echo "Error: No swap file path specified in configuration"
            return 1
        fi

        echo "Setting up swap with size: ${swap_size}GB at: $swap_file"

        if [ "$disable_zram" = "true" ]; then
            systemctl disable nvzramconfig
        fi

        # Create parent directory if it doesn't exist
        local swap_dir=$(dirname "$swap_file")
        if [ ! -d "$swap_dir" ]; then
            echo "Creating swap parent directory: $swap_dir"
            mkdir -p "$swap_dir"
        fi
        
        echo "Creating swap file..."
        if fallocate -l "${swap_size}G" "$swap_file" && \
           chmod 600 "$swap_file" && \
           mkswap "$swap_file" && \
           swapon "$swap_file"; then
            
            if ! grep -q "$swap_file" /etc/fstab; then
                echo "$swap_file  none  swap  sw 0  0" >> /etc/fstab
            fi
            
            yaml_set "$STATE_FILE" "swap.done" "true"
            yaml_set "$STATE_FILE" "swap.size" "$swap_size"
            yaml_set "$STATE_FILE" "swap.path" "$swap_file"
            return 0
        else
            echo "Failed to setup swap"
            return 1
        fi
    fi
    return 0
}

# Configure desktop GUI
setup_gui() {
    if [ "$(yaml_get "$STATE_FILE" "gui_disabled.done")" = "true" ]; then
        echo "GUI already configured, skipping..."
        return 0
    fi

    if should_run "gui_disabled" "Would you like to disable the desktop GUI on boot?"; then
        if systemctl set-default multi-user.target; then
            yaml_set "$STATE_FILE" "gui_disabled.done" "true"
            return 0
        else
            echo "Failed to configure GUI"
            return 1
        fi
    else
        systemctl set-default graphical.target
    fi
    return 0
}

# Configure Docker group
setup_docker_group() {
    if [ "$(yaml_get "$STATE_FILE" "docker_group.done")" = "true" ]; then
        echo "Docker group already configured, skipping..."
        return 0
    fi

    if should_run "docker_group" "Would you like to add $ORIGINAL_USER to the docker group?"; then
        if usermod -aG docker "$ORIGINAL_USER"; then
            yaml_set "$STATE_FILE" "docker_group.done" "true"
            return 0
        else
            echo "Failed to add user to Docker group"
            return 1
        fi
    fi
    return 0
}

# Configure power mode
setup_power_mode() {
    if [ "$(yaml_get "$STATE_FILE" "power_mode.done")" = "true" ]; then
        echo "Power mode already configured, skipping..."
        return 0
    fi

    if should_run "power_mode" "Would you like to set the power mode to 25W (recommended performance mode)?"; then
        local mode=$(yaml_get "$STATE_FILE" "power_mode.options.mode")
        if [ -z "$mode" ]; then
            mode="1"  # Default to 25W mode
        fi

        if nvpmodel -m "$mode"; then
            local mode_name=$(nvpmodel -q | grep "NV Power Mode" | cut -d':' -f2 | xargs)
            yaml_set "$STATE_FILE" "power_mode.done" "true"
            echo "Power mode set to: $mode_name"
            return 0
        else
            echo "Failed to set power mode"
            return 1
        fi
    fi
    return 0
}

# Ask for execution mode
ask_execution_mode() {
    echo "Choose execution mode:"
    echo "1) Run all configuration steps"
    echo "2) Select steps individually"
    while true; do
        read -p "Enter choice (1/2): " choice
        case $choice in
            1) 
                yaml_set "$STATE_FILE" "execution_mode" "all"
                return 0
                ;;
            2)
                yaml_set "$STATE_FILE" "execution_mode" "individual"
                return 0
                ;;
            *) echo "Please enter 1 or 2.";;
        esac
    done
}

# Check if step should be executed based on mode
should_execute_step() {
    local step=$1
    local prompt=$2
    local mode=$(yaml_get "$STATE_FILE" "execution_mode")
    
    if [ "$mode" = "all" ]; then
        return 0
    else
        ask_yes_no "Would you like to run the $step step?"
    fi
}

# Main execution
main() {
    echo "=== Jetson Setup Script ==="
    echo "This script will help configure your Jetson device."
    echo "You will need to reboot once after completion."
    echo

    check_permissions
    check_dependencies
    init_state
    detect_nvme_setup
    
    ask_execution_mode

    # NVMe Setup
    if should_execute_step "NVMe" "Configure NVMe drive"; then
        setup_nvme
    fi

    # Docker Runtime Setup
    if should_execute_step "Docker Runtime" "Configure Docker runtime"; then
        setup_docker_runtime
    fi

    # Docker Root Setup
    if should_execute_step "Docker Root" "Configure Docker root directory"; then
        setup_docker_root
    fi

    # Swap Setup
    if should_execute_step "Swap" "Configure swap space"; then
        setup_swap
    fi

    # GUI Setup
    if should_execute_step "GUI" "Configure desktop GUI"; then
        setup_gui
    fi

    # Docker Group Setup
    if should_execute_step "Docker Group" "Configure Docker group"; then
        setup_docker_group
    fi

    # Power Mode Setup
    if should_execute_step "Power Mode" "Configure power mode"; then
        setup_power_mode
    fi
    
    # Restart Docker service
    if should_execute_step "Docker Service" "Restart Docker service"; then
        systemctl restart docker
    fi

    echo
    echo "Configuration complete!"
    echo "Please reboot your system for all changes to take effect."
    if ask_yes_no "Would you like to reboot now?"; then
        reboot
    fi
}

# Run main function
main