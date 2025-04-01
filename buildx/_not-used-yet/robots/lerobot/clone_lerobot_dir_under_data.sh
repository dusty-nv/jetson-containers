#!/bin/bash

# Get the directory where the script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the relative target directory (3 levels up, then into "data" directory based on script location)
RELATIVE_TARGET_DIR="$SCRIPT_DIR/../../../data"

# Resolve the absolute path to the target directory
TARGET_DIR="$(realpath "$RELATIVE_TARGET_DIR")"

# Define the GitHub repository URL
REPO_URL="https://github.com/huggingface/lerobot.git"

# Extract the repo name from the URL
REPO_NAME=$(basename -s .git "$REPO_URL")

# Full path to the target directory where the repo will be cloned
CLONE_PATH="$TARGET_DIR/$REPO_NAME"

# Check if the directory already exists
if [ -d "$CLONE_PATH" ]; then
    echo "Directory $CLONE_PATH already exists. Skipping clone."
else
    # Clone the repository
    git clone "$REPO_URL" "$CLONE_PATH"

    # Check if cloning was successful by verifying if the directory exists
    if [ ! -d "$CLONE_PATH" ]; then
        echo "Error: Failed to clone repository to $CLONE_PATH"
        exit 1
    else
        echo "Repository cloned successfully to $CLONE_PATH."
    fi
fi