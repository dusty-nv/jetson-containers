#!/bin/bash

# Get the directory where the script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the overlay source directory (relative to where the script is located)
OVERLAY_DIR="$SCRIPT_DIR/lerobot_overlay"

# Define the relative target directory (3 levels up, then into "data/lerobot")
RELATIVE_TARGET_DIR="$SCRIPT_DIR/../../../data/lerobot"

# Resolve the absolute path to the target directory
TARGET_DIR="$(realpath "$RELATIVE_TARGET_DIR")"

# Check if the overlay directory exists
if [ ! -d "$OVERLAY_DIR" ]; then
    echo "Overlay directory $OVERLAY_DIR does not exist."
    exit 1
fi

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "[Error] Target directory $TARGET_DIR does not exist."
    echo " "
    echo "Run 'clone_lerobot_dir_under_data.sh' first."
    exit 1
fi

# Copy the contents from the overlay directory to the target directory
echo "Copying files from $OVERLAY_DIR to $TARGET_DIR"
cp -r "$OVERLAY_DIR"/* "$TARGET_DIR"

echo "Files copied successfully!"
