#!/bin/bash

GREEN="\033[92m"
RESET="\033[0m"

DEVICE_TYPE=$1
ENABLE_PAPI_PROFILING=${2:-OFF}  # If $3 is not provided, default to "OFF"

CONFIG_COMMAND=""
BUILD_COMMAND="cmake --build build --config Release -j $(nproc)"

if [ "$DEVICE_TYPE" == "gpu" ]; then
    CONFIG_COMMAND="cmake -B build -DGGML_CUDA=ON -DENABLE_PAPI=${ENABLE_PAPI_PROFILING}"
elif [ "$DEVICE_TYPE" == "cpu" ]; then
    CONFIG_COMMAND="cmake -B build -DGGML_CUDA=OFF -DENABLE_PAPI=${ENABLE_PAPI_PROFILING}"
else
    echo "Invalid device type. Use 'cpu' or 'gpu'."
    exit 1
fi

# Konfigurasyonu çalıştır
if eval "$CONFIG_COMMAND"; then
    echo -e "${GREEN}Configuration completed successfully${RESET}"
else
    echo "Configuration failed."
    exit 1
fi

# Derlemeyi çalıştır
if eval "$BUILD_COMMAND"; then
    echo -e "${GREEN}Build completed successfully${RESET}"
else
    echo "Build failed."
    exit 1
fi