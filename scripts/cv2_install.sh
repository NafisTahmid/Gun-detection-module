#!/bin/bash

# Function to detect the Jetson Nano version
get_jetson_nano_version() {
    # Extract just the major version (e.g., 32.4.4) from the full version string
    if dpkg -l | grep -q nvidia-l4t-core; then
        version=$(dpkg -l | grep nvidia-l4t-core | awk '{print $3}' | cut -d'-' -f1)
        echo "Detected Jetson Nano version: $version"
    else
        echo "NVIDIA Jetson packages not found. Exiting..."
        exit 1
    fi
}

# Function to determine compatible CUDA version and install necessary packages
install_cuda_dependencies() {
    local jetson_version=$1

    case $jetson_version in
        32.4.4)
            CUDA_VERSION="10-2"
            ;;
        32.5.1 | 32.6.1 | 32.6.3)
            CUDA_VERSION="10-2"
            ;;
        32.7.1 | 32.7.2 | 32.7.3 | 32.7.5)
            CUDA_VERSION="10-2"
            ;;
        *)
            echo "Unsupported Jetson Nano version: $jetson_version"
            exit 1
            ;;
    esac

    # Install CUDA dependencies based on the determined version
    echo "Installing CUDA $CUDA_VERSION dependencies..."
    sudo apt-get install -y cuda-toolkit-$CUDA_VERSION || { echo "Failed to install CUDA $CUDA_VERSION. Exiting..."; exit 1; }
}

# Check CUDA installation using nvcc
cuda_version=$(nvcc --version 2>/dev/null | grep -oP '(?<=release\s)\d+\.\d+' || echo "not installed")

# Ensure cuda_version is not "not installed"
if [ "$cuda_version" != "not installed" ]; then
    echo "CUDA version $cuda_version is installed."

    # Install dependencies for OpenCV
    sudo apt-get install -y build-essential cmake git python3-pip curl pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev || { echo "Failed to install dependencies. Exiting..."; exit 1; }

    exit 0
else
    echo "CUDA is not installed. Proceeding with installation of dependencies and OpenCV."
fi

# Set noninteractive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

# Remove existing OpenCV packages
sudo apt-get purge -y *libopencv* || { echo "Failed to remove existing OpenCV packages. Exiting..."; exit 1; }
echo "OpenCV packages removed."

# Update system
sudo apt update || { echo "Failed to update system. Exiting..."; exit 1; }
sudo apt upgrade -y || { echo "Failed to upgrade system. Exiting..."; exit 1; }

# Install common dependencies
sudo apt-get install -y build-essential cmake git python3-pip curl pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev || { echo "Failed to install dependencies. Exiting..."; exit 1; }
sudo apt-get install nvidia-l4t-core
# Detect Jetson Nano version and install CUDA dependencies
get_jetson_nano_version
install_cuda_dependencies "$version"

# Clone and build OpenCV 4.5.0
mkdir -p ~/opencv_build && cd ~/opencv_build || { echo "Failed to create or access opencv_build directory. Exiting..."; exit 1; }

if [ ! -d "opencv-4.5.0" ]; then
    wget https://github.com/opencv/opencv/archive/4.5.0.zip || { echo "Failed to download OpenCV 4.5.0. Exiting..."; exit 1; }
    unzip 4.5.0.zip && rm 4.5.0.zip || { echo "Failed to unzip OpenCV 4.5.0. Exiting..."; exit 1; }
fi

if [ ! -d "opencv_contrib-4.5.0" ]; then
    wget https://github.com/opencv/opencv_contrib/archive/4.5.0.zip || { echo "Failed to download OpenCV contrib modules. Exiting..."; exit 1; }
    unzip 4.5.0.zip && rm 4.5.0.zip || { echo "Failed to unzip OpenCV contrib modules. Exiting..."; exit 1; }
fi

# Build OpenCV
cd opencv-4.5.0 || { echo "Failed to change directory to opencv-4.5.0. Exiting..."; exit 1; }
mkdir -p build && cd build || { echo "Failed to create or access build directory. Exiting..."; exit 1; }

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib-4.5.0/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUBLAS=1 \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D CUDA_ARCH_BIN=7.2 \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
      -D PYTHON3_INCLUDE_DIR=/usr/include/python3.6m \
      -D PYTHON3_LIBRARY=/usr/lib/python3.6 \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
      -D BUILD_EXAMPLES=OFF \
      -D CMAKE_INSTALL_PREFIX=/usr/local .. || { echo "Failed to configure OpenCV build. Exiting..."; exit 1; }

make -j$(nproc) || { echo "Failed to build OpenCV. Exiting..."; exit 1; }
sudo make install || { echo "Failed to install OpenCV. Exiting..."; exit 1; }
sudo ldconfig || { echo "Failed to update library cache. Exiting..."; exit 1; }

# Cleanup
if [ -d ~/opencv_build ]; then
    rm -rf ~/opencv_build || { echo "Failed to remove opencv_build directory. Exiting..."; exit 1; }
    echo "Cleanup completed."
fi

echo "OpenCV 4.5.0 with CUDA and CuDNN has been installed."
