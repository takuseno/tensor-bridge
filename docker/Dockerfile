FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# this needs to avoid time zone question
ENV DEBIAN_FRONTEND=noninteractive

# install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        cmake \
        pkg-config \
        git \
        wget \
        unzip \
        unrar \
        python3-dev \
        python3-tk \
        python3-pip \
        zlib1g \
        zlib1g-dev \
        libgl1-mesa-dri \
        libgl1-mesa-glx \
        libglu1-mesa-dev \
        libasio-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        libosmesa6-dev \
        libglfw3 \
        libpcre3-dev \
        gfortran \
        libopenmpi-dev \
        gosu && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    pip install -U Cython==0.29.36 "jax[cuda12]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install nnabla nnabla-ext-cuda120

# create work directory
RUN mkdir /tensor-bridge
WORKDIR /tensor-bridge
