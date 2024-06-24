FROM docker.io/nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=root
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV QT_X11_NO_MITSHM 1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility,graphics

ENV PIP_ROOT_USER_ACTION=ignore
ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/${ROS_DISTRO}
ENV COLCON_PREFIX_PATH=/opt/ros/${ROS_DISTRO}
ENV LD_LIBRARY_PATH=/opt/ros/${ROS_DISTRO}/lib
ENV PATH=/opt/ros/${ROS_DISTRO}/bin:$PATH
ENV PYTHONPATH=/opt/ros/${ROS_DISTRO}/lib/python3.10/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2

RUN echo "Europe/London" > /etc/localtime ; echo "CUDA Version 12.5.0" > /usr/local/cuda/version.txt

#############################
# Install necessary libraries
#############################
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3 \
    gdb \
    git \
    git-core \
    bash-completion \
    openssh-client \
    python3-argcomplete \
    python3-pip \
    vim \
    lsb-release \
    wget \
    less \
    udev \
    zstd \
    sudo \
    libpng-dev \
    libgomp1 \
    python3-venv \
    file \
    libmagic-mgc \
    libmagic1 \
    software-properties-common \
    curl \
    bash-completion && \
    rm -rf /var/lib/apt/lists/*

###########################
# Install pip requirements
###########################
RUN pip3 install --upgrade pip==24.1
RUN pip3 install torch torchvision timm opencv-python transforms3d pyopengl setuptools==58.2.0 albumentations --no-binary albumentations,qudida

################
# STEREOLABS ZED
################
RUN wget -q -O ZED_SDK_Linux_Ubuntu22.run https://download.stereolabs.com/zedsdk/4.1/cu121/ubuntu22 && \
    chmod +x ZED_SDK_Linux_Ubuntu22.run ; ./ZED_SDK_Linux_Ubuntu22.run silent skip_cuda && \
    ln -sf /lib/x86_64-linux-gnu/libusb-1.0.so.0 /usr/lib/x86_64-linux-gnu/libusb-1.0.so && \
    rm ZED_SDK_Linux_Ubuntu22.run && \
    rm -rf /var/lib/apt/lists/*

# Make some tools happy
RUN mkdir -p /root/Documents/ZED/

######
# ROS 
######

RUN apt-get update -q && add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update -q && apt-get install --no-install-recommends -y \
    ros-dev-tools \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-grid-map-rviz-plugin \
    ros-${ROS_DISTRO}-ament-cmake-clang-format \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-nmea-msgs \
    ros-${ROS_DISTRO}-geographic-msgs \
    ros-${ROS_DISTRO}-diagnostic-updater \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-common-interfaces && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && \
    rosdep update --rosdistro $ROS_DISTRO

# COPY *neural_depth_2.0_2080.model /usr/local/zed/resources/neural_depth_2.0.model
# COPY *.8600_timing_cache /usr/local/zed/resources/.8600_timing_cache
# COPY *.neural_depth_2.0.model_optimized-fbcbl-1-75-12020-8801-8600-25-384-6144-48-64-512-72-0-0e53-512 /usr/local/zed/resources/.neural_depth_2.0.model_optimized-fbcbl-1-75-12020-8801-8600-25-384-6144-48-64-512-72-0-0e53-512
# COPY *SN21837864.conf /usr/local/zed/settings/


# Install gazebo
# RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
#   && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
#   && apt-get update && apt-get install -q -y --no-install-recommends \
#     ros-iron-gazebo* \
#   && rm -rf /var/lib/apt/lists/*
