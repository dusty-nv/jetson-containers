#!/usr/bin/env bash
# pass-through commands to 'docker run' with some defaults
# https://docs.docker.com/engine/reference/commandline/run/
ROOT="$(dirname "$(readlink -f "$0")")"

# Function to clean up background processes
cleanup() {
    if [[ ${#BG_PIDS[@]} -gt 0 ]]; then
        echo "Terminating background processes..."
        for pid in "${BG_PIDS[@]}"; do
            kill "$pid"  # Terminate each background process
            wait "$pid" 2>/dev/null  # Wait for the process to finish
        done
    fi
	sudo modprobe -r v4l2loopback
}

# Trap signals like INT (Ctrl+C) or TERM to invoke the cleanup function
trap cleanup INT TERM

# Initialize variables (default for arguments)
csi_to_webcam_conversion=false
capture_res="1640x1232@30"
output_res="1280x720@30"
capture_width="1640"
capture_height="1232"
capture_fps="30"
output_width="1280"
output_height="720"
output_fps="30"

# Loop through arguments
for arg in "$@"; do
	# Check for the --csi2webcam option
    if [[ "$arg" == "--csi2webcam" ]]; then
        csi_to_webcam_conversion=true
        continue  # Move to next argument
    fi

    # Check for --csi-capture-res
    if [[ "$arg" =~ --csi-capture-res= ]]; then
        csi_capture_res="${arg#*=}"
        # Extract width, height, and fps from capture_res
        if [[ $csi_capture_res =~ ([0-9]+)x([0-9]+)@([0-9]+) ]]; then
            capture_width="${BASH_REMATCH[1]}"
            capture_height="${BASH_REMATCH[2]}"
            capture_fps="${BASH_REMATCH[3]}"
        else
            echo "Invalid format for --csi-capture-res. Expected format: widthxheight@fps"
            exit 1
        fi
        continue
    fi

    # Check for --csi-output-res
    if [[ "$arg" =~ --csi-output-res= ]]; then
        csi_output_res="${arg#*=}"
        # Extract width, height, and fps from output_res
        if [[ $csi_output_res =~ ([0-9]+)x([0-9]+)@([0-9]+) ]]; then
            output_width="${BASH_REMATCH[1]}"
            output_height="${BASH_REMATCH[2]}"
            output_fps="${BASH_REMATCH[3]}"
        else
            echo "Invalid format for --csi-output-res. Expected format: widthxheight@fps"
            exit 1
        fi
        continue
    fi
done

# check for V4L2 devices
V4L2_DEVICES=""

if [[ "$csi_to_webcam_conversion" == true ]]; then

    echo "CSI to Webcam conversion enabled."
    echo "CSI Capture resolution: ${capture_width}x${capture_height}@${capture_fps}"
    echo "CSI Output resolution : ${output_width}x${output_height}@${output_fps}"

	# Check if v4l2loopback-dkms is installed
	if dpkg -l | grep -q v4l2loopback-dkms; then
		echo "( v4l2loopback-dkms is installed. )"
	else
		echo "[Error] v4l2loopback-dkms is not installed."
		echo " "
		echo "Perform the following command to first install v4l2loopback moddule."
		echo " "
		echo "    sudo apt update && sudo apt install v4l2loopback-dkms"
		echo " "
		exit 1
	fi

	# Check if v4l2-ctl is installed
	if command -v v4l2-ctl &> /dev/null
	then
		echo "(v4l2-ctl is installed)"
	else
		echo "[Error] v4l2-ctl is not installed"
		echo " "
		echo "Perform the following command to first install v4l-utils package."
		echo " "
		echo "    sudo apt install v4l-utils"
		echo " "
		exit 1
	fi

	sudo systemctl restart nvargus-daemon.service

	# Store /dev/video index number for each CSI camera found
	csi_indexes=()
	# Store /dev/video* device name for each CSI camera found
	csi_devices=()

	# Loop through all matching /dev/video* devices
	for device in /dev/video*; do
		# Use v4l2-ctl to check if the device supports RG10 (CSI camera format)
		if v4l2-ctl -d "$device" --list-formats-ext 2>/dev/null | grep -q "RG10"; then
			echo "$device is a CSI camera (RG10 format)"
			# Store the device name in array if CSI camera
			csi_devices+=("$device")
			# Extract the device index number and add to the csi_devices array
			dev_index=$(echo "$device" | grep -o '[0-9]\+')
			csi_indexes+=("$dev_index")
		else
			echo "$device is not a CSI camera (likely a webcam)"
			V4L2_DEVICES="$V4L2_DEVICES --device $device "
		fi
	done

	# Load the v4l2loopback module to create as many devices as CSI cameras found in the prior step
	sudo modprobe v4l2loopback devices=${#csi_indexes[@]} exclusive_caps=1 card_label="Cam1,Cam2"

	# Get all new /dev/video devices created by v4l2loopback
	new_devices=($(v4l2-ctl --list-devices | grep -A 1 "v4l2loopback" | grep '/dev/video' | awk '{print $1}'))
	echo "###### new_devices: ${new_devices[@]}"

	# add the created v4l2loopback devices
	if [[ -n "${new_devices[@]}" ]]; then
		for converted_device in ${new_devices[@]}; do
			V4L2_DEVICES="$V4L2_DEVICES --device $converted_device "
		done
	else
		echo "No v4l2loopback devices found."
	fi

	# Save the current DISPLAY variable
	ORIGINAL_DISPLAY=$DISPLAY

	# Start background processes for each CSI camera found
	i=0
	for csi_index in "${csi_indexes[@]}"; do
		echo "Starting background process for CSI camera device number: $csi_index"

		echo "CSI Capture resolution: ${capture_width}x${capture_height}@${capture_fps}"
		echo "CSI Output resolution : ${output_width}x${output_height}@${output_fps}"

		# Unset the DISPLAY env variable because, apparently, some GStreamer components might try to use this display
		# for video rendering or processing, which can conflict with other GStreamer elements or hardware device

		# Temporarily unset DISPLAY for the GStreamer command
		unset DISPLAY

		echo "gst-launch-1.0 -v nvarguscamerasrc sensor-id=${csi_index} \
					! 'video/x-raw(memory:NVMM), format=NV12, width=${capture_width}, height=${capture_height}, framerate=${capture_fps}/1' \
					! queue max-size-buffers=1 leaky=downstream ! nvvidconv \
					! 'video/x-raw, width=${output_width}, height=${output_height}, framerate=${output_fps}/1', format=I420 \
					! queue max-size-buffers=1 leaky=downstream ! nvjpegenc \
					! queue max-size-buffers=1 leaky=downstream ! multipartmux \
					! multipartdemux single-stream=1 \
					! \"image/jpeg, width=${output_width}, height=${output_height}, parsed=(boolean)true, colorimetry=(string)2:4:7:1, framerate=(fraction)${output_fps}/1, sof-marker=(int)0\" \
					! v4l2sink device=${new_devices[$i]} sync=false > $ROOT/logs/gst-launch-process_${csi_index}.txt 2>&1 &"
		gst-launch-1.0 -v nvarguscamerasrc sensor-id=${csi_index} \
					! "video/x-raw(memory:NVMM), format=NV12, width=${capture_width}, height=${capture_height}, framerate=${capture_fps}/1" \
					! queue max-size-buffers=1 leaky=downstream ! nvvidconv \
					! "video/x-raw, width=${output_width}, height=${output_height}, framerate=${output_fps}/1", format=I420 \
					! queue max-size-buffers=1 leaky=downstream ! nvjpegenc \
					! queue max-size-buffers=1 leaky=downstream ! multipartmux \
					! multipartdemux single-stream=1 \
					! "image/jpeg, width=${output_width}, height=${output_height}, parsed=(boolean)true, colorimetry=(string)2:4:7:1, framerate=(fraction)${output_fps}/1, sof-marker=(int)0" \
					! v4l2sink device=${new_devices[$i]} sync=false  > $ROOT/logs/gst-launch-process_${csi_index}.txt 2>&1 &

		# Store the PID of the background process if you want to manage it later
		BG_PIDS+=($!)
		echo "BG_PIDS: ${BG_PIDS[@]}"

		((i++))
	done

	# Restore the DISPLAY env variable
	export DISPLAY=$ORIGINAL_DISPLAY

else
	# Loop through all matching /dev/video* devices
	for device in /dev/video*; do
	    if [ -e "$device" ]; then  # Check if the device file exists
        	V4L2_DEVICES="$V4L2_DEVICES --device $device "
		fi
    done
fi

echo "V4L2_DEVICES: $V4L2_DEVICES"

if [ -n "$csi_indexes" ]; then
    echo "CSI_INDEXES:  $csi_indexes"
fi

# check for I2C devices
I2C_DEVICES=""

for i in {0..9}
do
	if [ -a "/dev/i2c-$i" ]; then
		I2C_DEVICES="$I2C_DEVICES --device /dev/i2c-$i "
	fi
done

# check for ttyACM devices
ACM_DEVICES=""

# Loop through all matching /dev/ttyACM* devices
for dev in /dev/ttyACM*; do
    if [ -e "$dev" ]; then  # Check if the device file exists
        ACM_DEVICES="$ACM_DEVICES --device $dev "
    fi
done

# check for display
DISPLAY_DEVICE=""

if [ -n "$DISPLAY" ]; then
	echo "### DISPLAY environmental variable is already set: \"$DISPLAY\""
	# give docker root user X11 permissions
	xhost +si:localuser:root || sudo xhost +si:localuser:root

	# enable SSH X11 forwarding inside container (https://stackoverflow.com/q/48235040)
	XAUTH=/tmp/.docker.xauth
	xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
	chmod 777 $XAUTH

	DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
fi

# check for jtop
JTOP_SOCKET=""
JTOP_SOCKET_FILE="/run/jtop.sock"

if [ -S "$JTOP_SOCKET_FILE" ]; then
	JTOP_SOCKET="-v /run/jtop.sock:/run/jtop.sock"
fi

# PulseAudio arguments
PULSE_AUDIO_ARGS=""

if [ -d "${XDG_RUNTIME_DIR}/pulse" ]; then
	PULSE_AUDIO_ARGS="-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native  -v ${XDG_RUNTIME_DIR}/pulse:${XDG_RUNTIME_DIR}/pulse"
fi

# SSH key for SCP uploads
SSH_KEY_VOLUME=""
SSH_KEY_ENV=""

if [ -n "$SCP_UPLOAD_KEY" ] && [ -f "$SCP_UPLOAD_KEY" ]; then
	# Mount SSH key to a standard location in the container
	# Mount to /root/.ssh/scp_upload_key and update the env var to point to it
	SSH_KEY_VOLUME="-v $SCP_UPLOAD_KEY:/root/.ssh/scp_upload_key:ro"
	SSH_KEY_ENV="-e SCP_UPLOAD_KEY=/root/.ssh/scp_upload_key"
fi

# extra flags
EXTRA_FLAGS=""

if [ -n "$HUGGINGFACE_TOKEN" ]; then
	EXTRA_FLAGS="$EXTRA_FLAGS --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN"
fi

# additional permission optional run arguments
OPTIONAL_PERMISSION_ARGS=""

if [ "$USE_OPTIONAL_PERMISSION_ARGS" = "true" ]; then
	OPTIONAL_PERMISSION_ARGS="-v /lib/modules:/lib/modules --device /dev/fuse --cap-add SYS_ADMIN --security-opt apparmor=unconfined"
fi

# check if sudo is needed
if [ $(id -u) -eq 0 ] || id -nG "$USER" | grep -qw "docker"; then
	SUDO=""
else
	SUDO="sudo"
fi

# Initialize an empty array for filtered arguments
filtered_args=()

# Loop through all provided arguments
for arg in "$@"; do
    if [[ "$arg" != "--csi2webcam" && "$arg" != --csi-capture-res=* && "$arg" != --csi-output-res=* ]]; then
        filtered_args+=("$arg")  # Add to the new array if not the argument to remove
    fi

    if [[ "$arg" = "--name" || "$arg" = --name* ]]; then
        HAS_CONTAINER_NAME=1
    fi
done

if [ -z "$HAS_CONTAINER_NAME" ]; then
    # Generate a unique container name so we can wait for it to exit and cleanup the bg processes after
    BUILD_DATE_TIME=$(date +%Y%m%d_%H%M%S)
    #CONTAINER_IMAGE_NAME=$(basename "${filtered_args[0]}")  # unfortunately this doesn't work in the general case, and you can't easily parse the container image from the command-line
    #SANITIZED_CONTAINER_IMAGE_NAME=$(echo "$CONTAINER_IMAGE_NAME" | sed 's/[^a-zA-Z0-9_.-]/_/g')
    CONTAINER_NAME="jetson_container_${BUILD_DATE_TIME}"
    CONTAINER_NAME_FLAGS="--name $CONTAINER_NAME"
fi

TEGRA="tegra"
if [ -z "${SYSTEM_ARCH}" ]; then
  ARCH=$(uname -m)

  if [ "$ARCH" = "aarch64" ]; then
	echo "### ARM64 architecture detected"
    if uname -a | grep -qi "$TEGRA"; then
      SYSTEM_ARCH="$TEGRA-$ARCH"
      echo "### Jetson Detected"
    else
      echo "### SBSA Detected"
      SYSTEM_ARCH="$ARCH"
    fi
  else
    echo "### x86 Detected"
    SYSTEM_ARCH="$ARCH"
  fi
fi

echo "SYSTEM_ARCH=$SYSTEM_ARCH"

if [ $SYSTEM_ARCH = "tegra-aarch64" ]; then
	# this file shows what Jetson board is running
	# /proc or /sys files aren't mountable into docker
	cat /proc/device-tree/model > /tmp/nv_jetson_model

    # https://stackoverflow.com/a/19226038
	( set -x ;

	$SUDO docker run --runtime nvidia --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -it --rm --network host \
		--shm-size=8g \
		--volume /tmp/argus_socket:/tmp/argus_socket \
		--volume /etc/enctune.conf:/etc/enctune.conf \
		--volume /etc/nv_tegra_release:/etc/nv_tegra_release \
		--volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
		--volume /var/run/dbus:/var/run/dbus \
		--volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
		--volume /var/run/docker.sock:/var/run/docker.sock \
		--volume $ROOT/data:/data \
		-v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro \
		--device /dev/snd \
		$PULSE_AUDIO_ARGS \
		--device /dev/bus/usb \
		$SSH_KEY_VOLUME $SSH_KEY_ENV \
		$OPTIONAL_PERMISSION_ARGS $DATA_VOLUME $DISPLAY_DEVICE $V4L2_DEVICES $I2C_DEVICES $ACM_DEVICES $JTOP_SOCKET $EXTRA_FLAGS \
		$CONTAINER_NAME_FLAGS \
		"${filtered_args[@]}"
	)

elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "x86_64" ]; then

	( set -x ;

	$SUDO docker run --gpus all -it --rm --network=host \
		--shm-size=8g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--env NVIDIA_DRIVER_CAPABILITIES=all \
		--volume $ROOT/data:/data \
		-v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro \
		--device /dev/snd \
		$PULSE_AUDIO_ARGS \
		$SSH_KEY_VOLUME $SSH_KEY_ENV \
		$OPTIONAL_ARGS $DATA_VOLUME $DISPLAY_DEVICE $V4L2_DEVICES $I2C_DEVICES $ACM_DEVICES $JTOP_SOCKET $EXTRA_FLAGS \
		$CONTAINER_NAME_FLAGS \
		"${filtered_args[@]}"
	)
fi

if [[ "$csi_to_webcam_conversion" == true ]]; then

	# Wait for the Docker container to finish (if it exits)
	docker wait "$CONTAINER_NAME"

	# When Docker container exits, cleanup will be called
	cleanup

fi
