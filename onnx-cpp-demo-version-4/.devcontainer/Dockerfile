FROM ubuntu:22.04

ARG ONNX_VERSION=1.19.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip ca-certificates pkg-config \
    libgomp1 gdb python3 python3-pip python3-venv curl \
  && rm -rf /var/lib/apt/lists/*

# Create workspace + extraction dir
RUN mkdir -p /workspace /opt/onnxruntime /workspace/python /workspace/models

# Install python packages needed to train & export Iris classifier
RUN python3 -m pip install --no-cache-dir \
    numpy pandas scikit-learn==1.7.2 onnx==1.14.0 skl2onnx==1.19.1 onnxruntime==1.19.0

# Download & extract ONNX Runtime NuGet package, copy headers & .so to standard locations
RUN wget -q "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ONNX_VERSION}" -O /tmp/onnxruntime.nupkg \
 && unzip -q /tmp/onnxruntime.nupkg -d /opt/onnxruntime \
 && if [ -d /opt/onnxruntime/runtimes/linux-x64/native ]; then cp -a /opt/onnxruntime/runtimes/linux-x64/native/libonnxruntime.so* /usr/local/lib/; fi \
 && if [ -d /opt/onnxruntime/native ]; then cp -a /opt/onnxruntime/native/libonnxruntime.so* /usr/local/lib/; fi \
 && if [ -d /opt/onnxruntime/build/native/include ]; then cp -r /opt/onnxruntime/build/native/include/* /usr/local/include/; fi \
 && if [ -d /opt/onnxruntime/include ]; then cp -r /opt/onnxruntime/include/* /usr/local/include/; fi \
 && ldconfig

VOLUME ["/workspace/src"]

ARG USERNAME=vscode
RUN useradd -m -s /bin/bash $USERNAME || true

USER $USERNAME
WORKDIR /workspace
