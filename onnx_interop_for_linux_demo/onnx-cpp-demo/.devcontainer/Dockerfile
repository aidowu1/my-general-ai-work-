# base Ubuntu (change to 24.04 if you prefer)
FROM ubuntu:22.04

ARG ONNX_VERSION=1.22.1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip ca-certificates pkg-config \
    libgomp1 gdb python3 python3-pip curl \
  && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /workspace /opt/onnxruntime

# Download & extract the ONNX Runtime NuGet package and copy headers/libs into system paths.
# Uses ONNX_VERSION arg; change the version there or via devcontainer.json if you want a different one.
RUN wget -q "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ONNX_VERSION}" -O /tmp/onnxruntime.nupkg \
 && unzip -q /tmp/onnxruntime.nupkg -d /opt/onnxruntime \
 # copy libraries (look in a few likely places)
 && if [ -d /opt/onnxruntime/runtimes/linux-x64/native ]; then cp -a /opt/onnxruntime/runtimes/linux-x64/native/libonnxruntime.so* /usr/local/lib/; fi \
 && if [ -d /opt/onnxruntime/native ]; then cp -a /opt/onnxruntime/native/libonnxruntime.so* /usr/local/lib/; fi \
 # copy headers from common locations present in NuGet packages
 && if [ -d /opt/onnxruntime/build/native/include ]; then cp -r /opt/onnxruntime/build/native/include/* /usr/local/include/; fi \
 && if [ -d /opt/onnxruntime/include ]; then cp -r /opt/onnxruntime/include/* /usr/local/include/; fi \
 && ldconfig

# Create a non-root user (optional, helpful for VS Code)
ARG USERNAME=vscode
RUN useradd -m -s /bin/bash $USERNAME || true
USER $USERNAME
WORKDIR /workspace

