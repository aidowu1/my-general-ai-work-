# .devcontainer/Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-venv \
    python3-pip \
    pkg-config \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME

# Ensure pip is recent
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install XGBoost Python wheel (which includes libxgboost)
# Note: this installs into the system python's site-packages.
RUN python3 -m pip install xgboost numpy pandas scikit-learn

# Keep locale sane
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Copy devcontainer helper and scripts (post create will run later)
COPY .devcontainer/postCreateCommand.sh /tmp/postCreateCommand.sh
RUN chmod +x /tmp/postCreateCommand.sh

USER $USERNAME
WORKDIR /workspace
