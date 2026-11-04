FROM ros:humble-ros-base

# Set noninteractive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]

# Setup locales
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

#Basics
RUN apt-get update && apt-get install -y \
    git wget curl gnupg2 lsb-release \
    build-essential cmake udev sudo \
    software-properties-common \
    iproute2 iputils-ping net-tools \
    clang-format \
    pre-commit \
    python3-rosdep \
    python3-vcstool \
    python3-colcon-common-extensions \
    ros-humble-diagnostic-updater \
    ros-humble-rviz2 \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    nano \
    vim \
    ros-humble-rqt* \
    ros-humble-example-interfaces \
    python3-pip \
    python3-venv \
    usbutils \
    libasound2-dev \
    alsa-utils \
    portaudio19-dev \
    x11-apps \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install critical Unitree ROS2 dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    libyaml-cpp-dev \
    git iputils-ping iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/unitreerobotics/unitree_ros2.git /opt/unitree_ros2 \
    && cd /opt/unitree_ros2/cyclonedds_ws \
    && source /opt/ros/humble/setup.bash \
    && colcon build

# Set up the environment middleware to talk to the G1
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=0

# Source everything in every shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /opt/unitree_ros2/cyclonedds_ws/install/setup.bash" >> /root/.bashrc \
    && echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> /root/.bashrc

WORKDIR /workspace