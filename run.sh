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

# Check for the --csi2webcam option
csi_to_webcam_conversion=false
for arg in "$@"; do
    if [[ "$arg" == "--csi2webcam" ]]; then
        csi_to_webcam_conversion=true
        break
    fi
done

# check for V4L2 devices
V4L2_DEVICES=""

if [[ "$csi_to_webcam_conversion" == true ]]; then

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
	sudo modprobe v4l2loopback devices=${#csi_indexes[@]} exclusive_caps=1 card_label="Converted from CSI camera"

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

	# Start background processes for each CSI camera found
	i=0
	for csi_index in "${csi_indexes[@]}"; do
		echo "Starting background process for CSI camera device number: $csi_index"

		# Run gst-launch-1.0 command in the background, suppressing all output
		echo "gst-launch-1.0 -v nvarguscamerasrc sensor-id=${csi_index} \
					! 'video/x-raw(memory:NVMM), format=NV12, width=1640, height=1232, framerate=30/1' \
					! nvvidconv \
					! 'video/x-raw, width=1600, height=1200, framerate=30/1', format=I420 \
					! nvjpegenc \
					! multipartmux \
					! multipartdemux single-stream=1 \
					! \"image/jpeg, width=1600, height=1200, parsed=(boolean)true, colorimetry=(string)2:4:7:1, framerate=(fraction)30/1, sof-marker=(int)0\" \
					! v4l2sink device=${new_devices[$i]} > /dev/null 2>&1 &"
		gst-launch-1.0 -v nvarguscamerasrc sensor-id=${csi_index} \
					! 'video/x-raw(memory:NVMM), format=NV12, width=1640, height=1232, framerate=30/1' \
					! nvvidconv \
					! 'video/x-raw, width=1600, height=1200, framerate=30/1', format=I420 \
					! nvjpegenc \
					! multipartmux \
					! multipartdemux single-stream=1 \
					! "image/jpeg, width=1600, height=1200, parsed=(boolean)true, colorimetry=(string)2:4:7:1, framerate=(fraction)30/1, sof-marker=(int)0" \
					! v4l2sink device=${new_devices[$i]} > /dev/null 2>&1 &
		# ping google.com > /dev/null 2>&1 &

		# Store the PID of the background process if you want to manage it later
		BG_PIDS+=($!)
		echo "BG_PIDS: ${BG_PIDS[@]}"

		((i++))
	done

else
	# Loop through all matching /dev/video* devices
	for device in /dev/video*; do
	    if [ -e "$device" ]; then  # Check if the device file exists
        	V4L2_DEVICES="$V4L2_DEVICES --device $device "
		fi
    done
fi

echo "V4L2_DEVICES: $V4L2_DEVICES"
echo "csi_indexes: $csi_indexes"

# check for I2C devices
I2C_DEVICES=""

for i in {0..9}
do
	if [ -a "/dev/i2c-$i" ]; then
		I2C_DEVICES="$I2C_DEVICES --device /dev/i2c-$i "
	fi
done

# check for display
DISPLAY_DEVICE=""

if [ -n "$DISPLAY" ]; then
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
    if [[ "$arg" != "--csi2webcam" ]]; then
        filtered_args+=("$arg")  # Add to the new array if not the argument to remove
    fi
done

# Track container ID for `docker wait`
container_id=""

# run the container
ARCH=$(uname -i)

if [ $ARCH = "aarch64" ]; then

	# this file shows what Jetson board is running
	# /proc or /sys files aren't mountable into docker
	cat /proc/device-tree/model > /tmp/nv_jetson_model

	set -x

	$SUDO docker run --runtime nvidia -it --rm --network host \
		--shm-size=8g \
		--volume /tmp/argus_socket:/tmp/argus_socket \
		--volume /etc/enctune.conf:/etc/enctune.conf \
		--volume /etc/nv_tegra_release:/etc/nv_tegra_release \
		--volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
		--volume /var/run/dbus:/var/run/dbus \
		--volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
		--volume /var/run/docker.sock:/var/run/docker.sock \
		--volume $ROOT/data:/data \
		--device /dev/snd \
		--device /dev/bus/usb \
		$OPTIONAL_PERMISSION_ARGS $DATA_VOLUME $DISPLAY_DEVICE $V4L2_DEVICES $I2C_DEVICES $JTOP_SOCKET $EXTRA_FLAGS \
		--name my_jetson_container \
		"${filtered_args[@]}"

	set +x

elif [ $ARCH = "x86_64" ]; then

	set -x

	$SUDO docker run --gpus all -it --rm --network=host \
		--shm-size=8g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--env NVIDIA_DRIVER_CAPABILITIES=all \
		--volume $ROOT/data:/data \
		$OPTIONAL_ARGS $DATA_VOLUME $DISPLAY_DEVICE $V4L2_DEVICES $I2C_DEVICES $JTOP_SOCKET $EXTRA_FLAGS \
		--name my_jetson_container \
		"${filtered_args[@]}"

	set +x
fi

if [[ "$csi_to_webcam_conversion" == true ]]; then

	# Wait for the Docker container to finish (if it exits)
	docker wait my_jetson_container

	# When Docker container exits, cleanup will be called
	cleanup

fi