# Stage 1: Builder - install Python packages
FROM ros:noetic-ros-base AS builder

# Basic build dependencies for Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build_python_packages

# Python packages (OpenCV, NumPy)
RUN pip3 install --no-cache-dir numpy opencv-python

# Clean up to reduce image size
RUN apt-get remove -y python3-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/build_python_packages

# Stage 2: Runtime container
FROM ros:noetic-ros-base

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    wget gpg \
    git \
    cmake \
    build-essential \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-camera-info-manager \
    ros-noetic-rviz \
    ros-noetic-tf2-ros \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*
    
    
RUN apt-get update && apt-get install -y \
    mesa-utils \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6

# Install librealsense
RUN git clone https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf librealsense
    
RUN apt-get update && apt-get install -y ros-noetic-diagnostic-updater

# Setup catkin workspace and build realsense-ros
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone -b ros1-legacy https://github.com/IntelRealSense/realsense-ros.git && \
    git clone https://github.com/pal-robotics/ddynamic_reconfigure.git && \
    cd /catkin_ws && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Set up environment for ROS + catkin workspace
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages

# Install VS Code for ARM64
RUN apt-get update && \
    apt-get install -y wget gpg && \
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg && \
    install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/ && \
    sh -c 'echo "deb [arch=arm64] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' && \
    apt-get update && \
    apt-get install -y code && \
    rm -rf /var/lib/apt/lists*

WORKDIR /app

CMD ["/bin/bash"]
