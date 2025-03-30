#!/bin/bash

# Function to display a countdown timer
countdown() {
  local seconds=$1
  while [ $seconds -gt 0 ]; do
    echo -ne "Launching dialog in $seconds seconds...\033[0K\r"
    sleep 1
    : $((seconds--))
  done
  echo -ne "\033[0K\r"  # Clear the line
}

# Function to check if dialog is installed
check_dialog() {
  if command -v dialog &> /dev/null; then
    echo "dialog is installed"
  else
    echo "dialog is not installed"
    read -p "dialog is not installed. Do you want to install it now? [y/N] " choice
    case "$choice" in 
      y|Y ) 
        if sudo -v; then
          sudo apt-get update
          sudo apt-get install dialog
        else
          echo "sudo access is required to install dialog. Please run the script with sudo."
          exit 1
        fi
        ;;
      * ) 
        echo "dialog is required for this script to run. Exiting."
        exit 1
        ;;
    esac
  fi
}

# Function to check the existence of required files for the Docker build
check_files() {
  local missing_files=()
  for file in "$@"; do
    if [ ! -f "$file" ]; then
      missing_files+=("$file")
    fi
  done

  if [ ${#missing_files[@]} -gt 0 ]; then
    echo "The following required files are missing:"
    for file in "${missing_files[@]}"; do
      echo " - $file"
    done
    exit 1
  fi
}

# Function to build a new Docker image
build_image() {
  # Check for required files
  check_files "buildx/verify.sh" "buildx/triton/build.sh" "buildx/triton/install.sh"

  # Get the current date and time for the new image tag
  NEW_TAG=$(date +"%Y-%m-%d-%H%M-%S")

  # Build the new Docker image
  docker buildx build \
    --builder mybuilder \
    --platform linux/arm64 \
    -t kairin/001:$NEW_TAG-1 \
    --build-arg BASE_IMAGE=kairin/001:nvcr.io-nvidia-pytorch-25.02-py3-igpu \
    --build-arg TRITON_VERSION=2.0.0 \
    --build-arg TRITON_BRANCH=main \
    --push \
    -f buildx/Dockerfile .

  # Update the script to use the new image tag
  sed -i "s|$(autotag kairin/001:.*)|$(autotag kairin/001:$NEW_TAG-1)|" $0

  # Pull the latest image
  docker pull kairin/001:$NEW_TAG-1

  echo "New image built and pulled: kairin/001:$NEW_TAG-1"
}

# Function to run the latest Docker image
run_image() {
  # List all available Docker images
  IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}")

  # Check if there are available images
  if [ -z "$IMAGES" ]; then
    echo "No available Docker images found."
    exit 1
  fi

  # Get the terminal window size
  HEIGHT=$(tput lines)
  WIDTH=$(tput cols)

  # Calculate dialog window size
  DIALOG_HEIGHT=$(( HEIGHT - 2 ))
  DIALOG_WIDTH=$(( WIDTH - 2 ))

  # Display the images in a dialog menu
  IMAGE=$(dialog --clear \
                 --backtitle "Select Docker Image" \
                 --title "Available Docker Images" \
                 --menu "Select an image to run" \
                 $DIALOG_HEIGHT $DIALOG_WIDTH $(( HEIGHT - 8 )) \
                 $(for img in $IMAGES; do echo $img $img; done) \
                 2>&1 >/dev/tty)

  # Run the selected Docker image
  if [ -n "$IMAGE" ]; then
    docker run --runtime nvidia -it --rm --network host --shm-size=8g --volume /tmp/argus_socket:/tmp/argus_socket --volume /etc/enctune.conf:/etc/enctune.conf --volume /etc/nv_tegra_release:/etc/nv_tegra_release --device /dev/video0:/dev/video0 $IMAGE
  else
    echo "No image selected. Exiting."
    exit 1
  fi
}

# Function to display menu options using dialog
menu() {
  CHOICE=$(dialog --clear \
                  --backtitle "Jetson Run Menu" \
                  --title "Main Menu" \
                  --menu "Use [UP/DOWN] keys to navigate and [Enter] to select" \
                  15 50 4 \
                  1 "Build a new Docker image" \
                  2 "Run the latest Docker image" \
                  3 "Quit" \
                  2>&1 >/dev/tty)

  case $CHOICE in
    1)
      build_image
      ;;
    2)
      run_image
      ;;
    3)
      exit 0
      ;;
    *)
      echo "Invalid option"
      ;;
  esac
}

# Main script
countdown 5
check_dialog
menu
