#!/bin/bash

# Run the script: ./valgrind.sh [number_of_cores] --debug

export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

# Check arguments
NUM_CORES=$(nproc)
DEBUG_MODE=0

for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    NUM_CORES=$arg
  elif [[ "$arg" == "--debug" ]]; then
    DEBUG_MODE=1
  fi
done

echo "Running with $NUM_CORES cores"
[ $DEBUG_MODE -eq 1 ] && echo "Debug mode: Valgrind enabled"

# Source files
SRCS=$(find src -name '*.cpp')

# Compiler and flags
CXX=g++-10
COMMON_FLAGS="-Wall -Wextra -march=native -std=c++20 -fopenmp"
PKG_FLAGS="$(pkg-config --cflags --libs opencv4)"
INCLUDE_FLAGS="-Iinclude"
EXTRA_LIBS="-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lfftw3 -lm"

# Optimization / debug flags
if [ $DEBUG_MODE -eq 1 ]; then
  OPT_FLAGS="-g -O0"  # Debug build
else
  OPT_FLAGS="-O3"
fi

# Build
echo "Compiling the program..."
$CXX $COMMON_FLAGS $OPT_FLAGS $INCLUDE_FLAGS $PKG_FLAGS $SRCS -o simulation_exec $EXTRA_LIBS

if [ $? -ne 0 ]; then
  echo "Compilation failed. Exiting."
  exit 1
fi

# Run (with valgrind if debug)
echo "Compilation successful. Launching simulation..."
if [ $DEBUG_MODE -eq 1 ]; then
  valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./simulation_exec $NUM_CORES
else
  ./simulation_exec $NUM_CORES
fi