#!/bin/bash

# Run the script: ./bottleneck.sh [number_of_cores]

export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [number_of_cores]"
  exit 1
fi

if [ "$#" -eq 1 ]; then
  NUM_CORES=$1
else
  NUM_CORES=$(nproc)
fi

echo "Running with $NUM_CORES cores"

SRCS=$(find src -name '*.cpp')

CXX=g++-10
OPT_FLAGS="-O3 -march=native -std=c++20 -fopenmp -pg -Wall -Wextra"
PKG_FLAGS="$(pkg-config --cflags --libs opencv4)"
INCLUDE_FLAGS="-Iinclude"
EXTRA_LIBS="-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lfftw3 -lm"

# Clean previous profiling data
rm -f gmon.out analysis.txt simulation_exec

echo "Compiling with profiling and optimizations..."
$CXX $OPT_FLAGS $INCLUDE_FLAGS $PKG_FLAGS $SRCS -o simulation_exec $EXTRA_LIBS

if [ $? -ne 0 ]; then
  echo "Compilation failed. Exiting."
  exit 1
fi

echo "Running simulation (this will generate gmon.out)..."
time ./simulation_exec $NUM_CORES

echo "Generating gprof report..."
gprof ./simulation_exec gmon.out > analysis.txt

echo ""
echo "Profiling complete. Report saved in analysis.txt"
echo "You can view it with:"
echo "  less analysis.txt"
