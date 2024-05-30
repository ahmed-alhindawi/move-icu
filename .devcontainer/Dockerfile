FROM docker.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

#############################
# Install necessary libraries
#############################
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
  bash-completion \
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
  && rm -rf /var/lib/apt/lists/*

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV VIRTUAL_ENV=/home/$USERNAME/venv

ENV DEBIAN_FRONTEND=noninteractive

ENV QT_X11_NO_MITSHM 1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility,graphics


ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/${ROS_DISTRO}
ENV COLCON_PREFIX_PATH=/opt/ros/${ROS_DISTRO}
ENV LD_LIBRARY_PATH=/opt/ros/${ROS_DISTRO}/lib
ENV PATH=/opt/ros/${ROS_DISTRO}/bin:$PATH
ENV PYTHONPATH=/opt/ros/${ROS_DISTRO}/lib/python3.10/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2

RUN echo "Europe/London" > /etc/localtime ; echo "CUDA Version 12.2.0" > /usr/local/cuda/version.txt

###################
# CREATE LOCAL USER
###################
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install --upgrade pip

#####
# ROS 
#####

RUN sudo apt-get update -q && sudo add-apt-repository universe \
  && sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && sudo echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && sudo rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update -qq && sudo apt-get install -y \
  ros-dev-tools \
  ros-${ROS_DISTRO}-desktop \
  ros-${ROS_DISTRO}-xacro \
  ros-${ROS_DISTRO}-ament-cmake-clang-format \
  ros-${ROS_DISTRO}-grid-map-rviz-plugin \
  ros-${ROS_DISTRO}-nmea-msgs \
  ros-${ROS_DISTRO}-geographic-msgs \
  ros-${ROS_DISTRO}-diagnostic-updater \
  ros-${ROS_DISTRO}-robot-localization \
  && sudo rm -rf /var/lib/apt/lists/*

RUN sudo rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO


################
# StereoLabs ZED
################

WORKDIR /tmp/
RUN wget -q -O ZED_SDK_Linux.run https://download.stereolabs.com/zedsdk/4.1/cu121/ubuntu22 && \
    sudo chmod +x ZED_SDK_Linux.run ; sudo ./ZED_SDK_Linux.run silent skip_cuda skip_python && \
    sudo ln -sf /lib/x86_64-linux-gnu/libusb-1.0.so.0 /usr/lib/x86_64-linux-gnu/libusb-1.0.so && \
    rm ZED_SDK_Linux.run && \
    sudo rm -rf /var/lib/apt/lists/*

# Make some tools happy
RUN mkdir -p /home/$USERNAME/Documents/ZED/

##########################
# Install pip requirements
##########################
COPY requirements.txt .
RUN pip3 install -r requirements.txt

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

COPY ./ros_entrypoint.sh /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

