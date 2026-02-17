#!/usr/bin/env bash

set -exo pipefail

# INPUT:
# ```
# repo_url_1 repo_url_2 repo_url_3,...
# ```

ROOT="/opt/stable-diffusion-webui"
PYTHONPATH="$ROOT"
extensions_dir="$ROOT/extensions-builtin"

# builtin_extensions extensions won't be removed
builtin_extensions=(
    "extra-options-section",
    "hypertile",
    "prompt-bracket-checker",
    "canvas-zoom-and-pan",
    "mobile",
    "LDSR",
    "Lora",
    "SwinIR",
    "ScuNET"
)

green="\033[1;32m"
red="\033[1;31m"
yellow="\033[1;33m"
suffix="\033[0m"

function log() {
    local color="$1"
    local message="$2"

    case "$color" in
        "green")
            echo -e "${green}${message}${suffix}"
            ;;
        "red")
            echo -e "${red}${message}${suffix}"
            ;;
        "yellow")
            echo -e "${yellow}${message}${suffix}"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

function list_directories() {
    local target_directory="$1"

    if [ -d "$target_directory" ]; then
        find "$target_directory" -mindepth 1 -maxdepth 1 -type d
    else
        echo "Error: $target_directory is not a valid directory." >&2
        return 1
    fi
}

function install() {
    local repository_url="$1"
    local repository_name=$(echo "$repository_url" | rev | cut -d'/' -f1 | rev)
    local repository_path="$extensions_dir/$repository_name"
    local repository_install_script_path="$repository_path/install.py"
    local repository_requirements_file="$repository_path/requirements.txt"

    # remove all already installed dependencies before installation using install.py
    # (sometimes install.py uses requirements.txt file as a source)
    if [ -f "$repository_requirements_file" ]; then
        sed 's|^torch.*|torch|g' -i "$repository_requirements_file"
        sed 's|^torchvision.*|torchvision|g' -i "$repository_requirements_file"
        sed 's|^onnx.*|onnx|g' -i "$repository_requirements_file"
        sed 's|^onnxruntime.*|onnxruntime|g' -i "$repository_requirements_file"
        sed 's|^numpy.*|numpy|g' -i "$repository_requirements_file"
        sed 's|^opencv-python.*|opencv-python|g' -i "$repository_requirements_file"
    fi

    # ModuleNotFoundError: No module named 'launch'
    if [ -f "$repository_install_script_path" ]; then
        PYTHONPATH=${ROOT} python3 "$repository_install_script_path"
        log green "[INSTALL] ✅ Install Script of $repository_name is completed!"
    else
        log red "[INSTALL] Install Script install.py not found for $repository_name in $repository_path..."
    fi
}

function clone() {
    local repository_url="$1"
    local repository_name=$(echo "$repository_url" | rev | cut -d'/' -f1 | rev)
    local repository_path="$extensions_dir/$repository_name"

    log green "[GIT] Cloning $repository_name to $repository_path..."
    git clone --jobs 0 "$repository_url" "$repository_path"
}

function upgrade() {
    local repository_url="$1"
    local repository_name=$(echo "$repository_url" | rev | cut -d'/' -f1 | rev)
    local repository_path="$extensions_dir/$repository_name"

    if [ -d "$repository_path" ]; then
        log yellow "[UPGRADE] Extension $repository_name is already installed in $repository_path, performing upgrade!"
        git -C "$repository_path" pull
        install "$repository_url"
        log green "[UPGRADE] ✅ Upgrade of $repository_name is completed!"
    fi
}

function remove() {
    local repository_name="$1"
    local repository_path="$extensions_dir/$repository_name"

    if [ -d "$repository_path" ]; then
        rm -rf "$repository_path"
        log yellow "[REMOVED] ✅ Removed previous version of $repository_name..."
    fi
}

function maybe_remove() {
    local repository_url="$1"
    local repository_name=$(echo "$repository_url" | rev | cut -d'/' -f1 | rev)

    remove "$repository_name"
}

function compare_directories() {
    local new_extensions=("$@")
    local installed_extensions_paths=($(list_directories "$extensions_dir"))

    # gather installed extensions names
    installed_extensions_names=()
    for item in "${installed_extensions_paths[@]}"; do
        name=$(echo "$item" | rev | cut -d'/' -f1 | rev)
        installed_extensions_names+=("$name")
    done

    # gather new (incoming) extensions names
    incoming_extensions_names=()
    for item in "${new_extensions[@]}"; do
        name=$(echo "$item" | rev | cut -d'/' -f1 | rev)
        incoming_extensions_names+=("$name")
    done

    # remove extensions if not needed
    for extension in "${installed_extensions_names[@]}"; do
        # Check if the extension is not present in incoming_extensions_names
        if [[ ! "${incoming_extensions_names[@]}" =~ "$extension" ]] && \
           [[ ! "${builtin_extensions[@]}" =~ "$extension" ]]; then
            # If not present then remove it
            log red "[UNINSTALL] $extension..."
            remove "$extension"
        else
            log green "[BUILT-IN] $extension"
        fi
    done
}

compare_directories "$@"

# Loop over all the repository URLs passed as arguments
for repository_url in "$@"; do
    repository_url=$(echo "${repository_url}" | tr -d '[:space:]')
    if [ -n "${repository_url}" ]; then
        maybe_remove "$repository_url"
        clone "$repository_url"
        install "$repository_url"
    fi
done
