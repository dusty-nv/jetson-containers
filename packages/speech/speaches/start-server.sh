#!/bin/bash
# https://stackoverflow.com/a/4319666
shopt -s huponexit

# Download model files if they don't exist
model_dir="/data/models/huggingface/models--hexgrad--Kokoro-82M/snapshots/main"
mkdir -p "$model_dir"

model_files=(
    "kokoro-v0_19.onnx"
    "voices.bin"
)

for file in "${model_files[@]}"; do
    file_path="$model_dir/$file"
    if [ -f "$file_path" ] && [ -s "$file_path" ]; then
        echo "Model file $file already exists, skipping download"
        continue
    fi

    echo "Downloading $file..."
    for attempt in {1..5}; do
        if wget -q --show-progress --progress=bar:force:noscroll -O "$file_path" "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/$file"; then
            echo "Successfully downloaded $file"
            break
        else
            echo "Attempt $attempt failed to download $file"
            if [ $attempt -lt 5 ]; then
                echo "Retrying in 10 seconds..."
                sleep 10
            else
                echo "Failed to download $file after 5 attempts"
                exit 1
            fi
        fi
    done
done

SPEACHES_DEFAULT_CMD="python3 -m uvicorn speaches.main:create_app --host 0.0.0.0 --port $PORT --factory"
SPEACHES_STARTUP_LAG=5

printf "Starting Speaches server:\n\n"
printf "  ${SPEACHES_DEFAULT_CMD}\n\n"

if [ "$#" -gt 0 ]; then
    ${SPEACHES_DEFAULT_CMD} 2>&1 &
    echo ""
    sleep ${SPEACHES_STARTUP_LAG}
    echo "Running command:  $@"
    echo ""
    sleep 1
    "$@"
else
    ${SPEACHES_DEFAULT_CMD}
fi
