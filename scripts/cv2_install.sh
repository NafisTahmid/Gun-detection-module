#!/bin/bash

if python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" > /dev/null 2>&1; then
    echo "OpenCV with CUDA support is already installed for Python 3."
    sudo apt-get install -y build-essential cmake git python3-pip curl pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev || { echo "Failed to install dependencies. Exiting..."; exit 1; }
    exit 0
fi

# Set noninteractive mode for apt-get
export DEBIAN_FRONTEND=noninteractive

sudo sudo apt-get purge *libopencv*

# Update system
sudo apt update || { echo "Failed to update system. Exiting..."; exit 1; }
sudo apt upgrade -y || { echo "Failed to upgrade system. Exiting..."; exit 1; }

# Install dependencies

# Install CUDA and CuDNN - Follow NVIDIA's official documentation

# Clone/OpenCV 4.5.0
mkdir -p ~/opencv_build && cd ~/opencv_build
if [ ! -d "opencv-4.5.0" ]; then
    wget https://github.com/opencv/opencv/archive/4.5.0.zip
    unzip 4.5.0.zip && rm 4.5.0.zip
fi

# Clone/OpenCV Contrib 4.5.0
if [ ! -d "opencv_contrib-4.5.0" ]; then
    wget https://github.com/opencv/opencv_contrib/archive/4.5.0.zip
    unzip 4.5.0.zip && rm 4.5.0.zip
fi

# Build OpenCV
cd opencv-4.5.0
mkdir -p build && cd build

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
      -D CMAKE_INSTALL_PREFIX=/usr/local ..

make -j$(nproc)
sudo make install
sudo ldconfig

# Cleanup
if [ -d ~/opencv_build ]; then
    rm -r ~/opencv_build
    echo "Cleanup completed."
fi

echo "OpenCV 4.5.0 with CUDA and CuDNN has been installed."
