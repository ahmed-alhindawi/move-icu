## Use an official CUDA image as the base image
FROM docker.io/nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set environment variables for CUDA
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc-11 \
    g++-11 \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    libffi-dev \
    libssl-dev \
    && apt-get clean

# Set GCC and G++ alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 10

# Create a working directory in the container
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt /workspace/requirements.txt

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install -r /workspace/requirements.txt

# Copy the rest of your project files into the container
COPY . /workspace



