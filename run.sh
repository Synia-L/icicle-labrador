#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
show_help() {
  echo "Usage: $0 [-d DEVICE_TYPE] [-b BACKEND_INSTALL_DIR] [-c] [-m] [-v] [-o]"
  echo
  echo "Options:"
  echo "  -d DEVICE_TYPE            Specify the device type (default: CPU)"
  echo "  -b BACKEND_INSTALL_DIR    Specify the backend installation directory (default: empty)"
  echo "  -c                        Use custom NTT implementation (CUDA only)"
  echo "  -m                        Use custom MatMul implementation (CUDA only)"
  echo "  -v                        Use custom VecOps implementation (CUDA only)"
  echo "  -o                        Use custom Misc Ops implementation (CUDA only, decompose/recompose/jl_projection/matrix_transpose)"
  echo "  -h                        Show this help message"
  exit 0
}

# Parse command line options
USE_CUSTOM_NTT=false
USE_CUSTOM_MATMUL=false
USE_CUSTOM_VEC_OPS=false
USE_CUSTOM_MISC_OPS=false
while getopts ":d:b:cmvoh" opt; do
  case ${opt} in
    d )
      DEVICE_TYPE=$OPTARG
      ;;
    b )
      ICICLE_BACKEND_INSTALL_DIR="$(realpath ${OPTARG})"
      ;;
    c )
      USE_CUSTOM_NTT=true
      ;;
    m )
      USE_CUSTOM_MATMUL=true
      ;;
    v )
      USE_CUSTOM_VEC_OPS=true
      ;;
    o )
      USE_CUSTOM_MISC_OPS=true
      ;;
    h )
      show_help
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      show_help
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument" 1>&2
      show_help
      ;;
  esac
done

# Set default values if not provided
: "${DEVICE_TYPE:=CPU}"
: "${ICICLE_BACKEND_INSTALL_DIR:=}"

DEVICE_TYPE_LOWERCASE=$(echo "$DEVICE_TYPE" | tr '[:upper:]' '[:lower:]')

# Create necessary directories
mkdir -p build/src
mkdir -p build/icicle

# Paths from root directory
ICILE_DIR=$(realpath "icicle/")
ICICLE_BACKEND_SOURCE_DIR="${ICILE_DIR}/backend/${DEVICE_TYPE_LOWERCASE}"

# Download and extract Icicle release for non-CPU devices if backend not specified
if [ "$DEVICE_TYPE" != "CPU" ] && [ -z "${ICICLE_BACKEND_INSTALL_DIR}" ]; then
  echo "Downloading Icicle release for ${DEVICE_TYPE}"
  mkdir -p build/icicle_release
  
  # Check if extracted directory exists, not just if tar.gz exists
  if [ ! -d "build/icicle_release/icicle" ]; then
    if [ ! -f "build/icicle_release/icicle.tar.gz" ]; then
      wget -q https://github.com/ingonyama-zk/icicle/releases/download/v4.0.0/icicle_4_0_0-ubuntu22-cuda122.tar.gz -O build/icicle_release/icicle.tar.gz
    fi
    echo "Extracting Icicle release..."
    tar -xzf build/icicle_release/icicle.tar.gz -C build/icicle_release
  else
    echo "Icicle release already extracted in build/icicle_release, skipping."
  fi
  
  export ICICLE_BACKEND_INSTALL_DIR=$(realpath "build/icicle_release/icicle")
  echo "Using downloaded Icicle release at ${ICICLE_BACKEND_INSTALL_DIR}"
  export LD_LIBRARY_PATH="${ICICLE_BACKEND_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
  
  # Build only the src project when using pre-built release
  cmake -DCMAKE_BUILD_TYPE=Release -S "${ICILE_DIR}" -B build/icicle -DRING=babykoala
  cmake --build build/icicle -j
  cmake -DCMAKE_BUILD_TYPE=Release -S src -B build/src
  cmake --build build/src -j
# Build Icicle backend locally if source exists and no backend install dir specified
elif [ "$DEVICE_TYPE" != "CPU" ] && [ ! -d "${ICICLE_BACKEND_INSTALL_DIR}" ] && [ -d "${ICICLE_BACKEND_SOURCE_DIR}" ]; then
  echo "Building icicle and ${DEVICE_TYPE} backend"
  cmake -DCMAKE_BUILD_TYPE=Release -DRING=babykoala "-D${DEVICE_TYPE}_BACKEND"=local -S "${ICILE_DIR}" -B build/icicle
  export ICICLE_BACKEND_INSTALL_DIR=$(realpath "build/icicle/backend")
  
  # Build both icicle and src
  cmake --build build/icicle -j
  cmake -DCMAKE_BUILD_TYPE=Release -S src -B build/src
  cmake --build build/src -j
else
  echo "Building icicle without backend, ICICLE_BACKEND_INSTALL_DIR=${ICICLE_BACKEND_INSTALL_DIR}"
  export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR}"
  cmake -DCMAKE_BUILD_TYPE=Release -DRING=babykoala -S "${ICILE_DIR}" -B build/icicle
  
  # Build both icicle and src
  cmake --build build/icicle -j
  cmake -DCMAKE_BUILD_TYPE=Release -S src -B build/src
  cmake --build build/src -j
fi

# Run the example
if [ "$DEVICE_TYPE" = "CUDA" ]; then
  CUSTOM_FLAGS=""
  if [ "$USE_CUSTOM_NTT" = true ]; then
    CUSTOM_FLAGS="$CUSTOM_FLAGS --custom-ntt"
    echo "========================================="
    echo "CUSTOM NTT ENABLED"
    echo "========================================="
  fi
  if [ "$USE_CUSTOM_MATMUL" = true ]; then
    CUSTOM_FLAGS="$CUSTOM_FLAGS --custom-matmul"
    echo "========================================="
    echo "CUSTOM MATMUL ENABLED"
    echo "========================================="
  fi
  if [ "$USE_CUSTOM_VEC_OPS" = true ]; then
    CUSTOM_FLAGS="$CUSTOM_FLAGS --custom-vec-ops"
    echo "========================================="
    echo "CUSTOM VEC OPS ENABLED"
    echo "========================================="
  fi
  if [ "$USE_CUSTOM_MISC_OPS" = true ]; then
    CUSTOM_FLAGS="$CUSTOM_FLAGS --custom-misc-ops"
    echo "========================================="
    echo "CUSTOM MISC OPS ENABLED"
    echo "========================================="
  fi
  
  if [ -n "$CUSTOM_FLAGS" ]; then
    echo ""
    ./build/src/example "$DEVICE_TYPE" $CUSTOM_FLAGS
  else
    ./build/src/example "$DEVICE_TYPE"
  fi
else
  ./build/src/example "$DEVICE_TYPE"
fi

# Optional: Memory check (uncomment to use)
# compute-sanitizer --tool memcheck --leak-check full ./build/src/example "$DEVICE_TYPE" > sanitizer_output.log 2>&1