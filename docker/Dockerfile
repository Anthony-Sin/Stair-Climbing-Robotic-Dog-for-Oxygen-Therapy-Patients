FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (excluding python3-opencv)
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    libgl1-mesa-glx libglfw3-dev libusb-1.0-0-dev \
    libssl-dev libffi-dev cmake git wget unzip ffmpeg \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV build dependencies
RUN apt-get update && apt-get install -y \
    build-essential pkg-config \
    libjpeg-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk2.0-dev libcanberra-gtk-module \
    libxvidcore-dev libx264-dev libgtk-3-dev \
    libtbb2 libtbb-dev libdc1394-22-dev \
    libv4l-dev v4l-utils \
    libopenblas-dev libatlas-base-dev libblas-dev \
    liblapack-dev gfortran libhdf5-dev \
    protobuf-compiler libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----------------------
# Install pyrealsense2
# ----------------------
RUN pip3 install --no-cache-dir pyrealsense2

# ----------------------
# Build OpenCV with CUDA support for Jetson
# ----------------------
WORKDIR /opt
RUN git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv_contrib.git

WORKDIR /opt/opencv
RUN mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_CUDA=ON \
          -D CUDA_ARCH_BIN="8.7,8.9,9.0" \
          -D CUDA_ARCH_PTX="" \
          -D WITH_CUBLAS=1 \
          -D WITH_CUFFT=ON \
          -D WITH_CURAND=ON \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=ON \
          -D WITH_GSTREAMER=ON \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D INSTALL_C_EXAMPLES=OFF \
          -D PYTHON_EXECUTABLE=$(which python3) \
          -D PYTHON3_EXECUTABLE=$(which python3) \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Clean up OpenCV source to save space
RUN rm -rf /opt/opencv /opt/opencv_contrib

# ----------------------
# Install Python packages (NO opencv-python here)
# ----------------------
# NOTE: TensorRT is provided by the l4t-jetpack base at /usr/src/tensorrt, matched to the
# on-device CUDA/cuDNN. Do NOT pip-install a different TensorRT: the old
# `tensorrt==10.7.0` from pypi.nvidia.com overrode the JetPack TRT (8.6) that BUILT the
# engines, so deserialize_cuda_engine() returned None at runtime with no actionable error
# (review §5). Use the base image's TRT.
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pillow \
    scipy \
    tqdm \
    onnx onnxruntime

# ----------------------
# Clone and install Ultralytics (YOLOv11) with no OpenCV deps
# ----------------------
RUN git clone https://github.com/ultralytics/ultralytics.git /workspace/ultralytics
WORKDIR /workspace/ultralytics
RUN pip3 install --no-cache-dir . --no-deps

# Install CLIP for YOLO-World open-vocabulary classification
RUN pip3 install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# Manually install all Ultralytics dependencies except opencv-python
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    tqdm \
    scipy \
    pyyaml \
    pandas \
    thop \
    seaborn \
    requests \
    psutil \
    tensorboard \
    networkx==2.8.8 \
    --extra-index-url https://pypi.nvidia.com

# PyTorch/torchvision for Jetson (aarch64). The old `--index-url .../whl/cu124` is the x86
# CUDA index -- on the Jetson it resolves to a CPU-only (or no) wheel, so torch silently ran
# on the CPU and inference fell off a cliff vs the 50 Hz loop (review §5). Install the NVIDIA
# Jetson build instead. VERIFY the index tag matches the JetPack/CUDA on THIS base
# (l4t-jetpack r36.2 -> CUDA 12.2); if the tag is wrong the build FAILS loudly (good) rather
# than silently installing CPU torch. Pin exact torch/torchvision versions once verified.
RUN pip3 install --no-cache-dir torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu122

RUN pip install --no-cache-dir numpy==1.23.5 --force-reinstall

# Clone and install ByteTrack from source
# Install ByteTrack (minimal install without clobbering existing packages)
RUN git clone https://github.com/ifzhang/ByteTrack.git /opt/bytetrack && \
    pip install --no-cache-dir --no-deps -e /opt/bytetrack
# Install additional python packages required by ByteTrack
RUN pip install --no-cache-dir loguru lap cython_bbox

# Install CycloneDDS 0.10.2 and set it up properly
WORKDIR /opt
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds.git -b releases/0.10.x && \
    cd cyclonedds && \
    mkdir build install && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install

ENV CYCLONEDDS_HOME=/opt/cyclonedds/install
ENV CMAKE_PREFIX_PATH=${CYCLONEDDS_HOME}:${CMAKE_PREFIX_PATH}

# Install CycloneDDS Python bindings (must match C library version)
RUN pip install --no-cache-dir cyclonedds==0.10.2

# Install unitree_sdk2_python from source without its dependencies
WORKDIR /opt
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2_python && \
    pip install --no-cache-dir --no-deps -e .

# Manually install other dependencies for unitree_sdk2_python
RUN pip install --no-cache-dir numpy

# ----------------------
# Install Unitree Z1 SDK (C++ and Python bindings) under /opt
# ----------------------
RUN apt-get update && apt-get install -y libeigen3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/unitreerobotics/z1_sdk.git

RUN pip install --no-cache-dir pybind11

# Build the SDK including Python bindings (CMake will find pybind11)
WORKDIR /opt/z1_sdk
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# To inspect the tensorrt engine
RUN pip install --no-cache-dir polygraphy colored

# No setup.py available, so just add the built module to Python path
ENV PYTHONPATH="/opt/z1_sdk/build/examples_py:${PYTHONPATH}"
# After installing TensorRT (or confirming it's already in /usr/src/tensorrt)
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

# Make sure unitree_sdk2py is always on the Python path
ENV PYTHONPATH="/opt/unitree_sdk2_python:${PYTHONPATH}"

RUN pip install --no-cache-dir kornia kornia-rs --no-deps
RUN pip install --no-cache-dir ecs-logging

# ----------------------
# On-frame HUD (core/hud): real GPU-rendered overlay via arcv (ModernGL vector/
# text batches + HDR bloom). Container already has --gpus all --runtime nvidia
# (docker_run.sh) so a standalone moderngl context should have EGL/GL access;
# NOT YET VERIFIED on real Jetson Orin hardware -- check for a moderngl context
# creation error on first boot after this image change.
#
# arcv's own pyproject.toml has a packaging bug: `packages = ["arcv"]` already
# bundles arcv/shaders + arcv/resources by default, and the extra
# [tool.hatch.build.targets.wheel.force-include] block re-adds the same paths,
# so `pip install git+https://...` fails wheel-building on a duplicate archive
# entry (confirmed). Strip that redundant trailing block before building --
# the shader/font assets still end up in the wheel via the default packages
# inclusion (verified: 84 files incl. all .frag/.vert/.ttf).
# ----------------------
RUN pip install --no-cache-dir "moderngl>=5.10" \
    && git clone --depth 1 https://github.com/Anthony-Sin/arcv.git /opt/arcv_build \
    && sed -i '/\[tool\.hatch\.build\.targets\.wheel\.force-include\]/,$d' /opt/arcv_build/pyproject.toml \
    && pip install --no-cache-dir --no-deps /opt/arcv_build \
    && rm -rf /opt/arcv_build

# Clean up torch wheel and pip cache (safe final cleanup)
RUN rm -rf /root/.cache/pip

# Default to workspace directory
WORKDIR /workspace
CMD ["bash"]
