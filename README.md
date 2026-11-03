# Artifact for "EnzymeMLIR: Combining Domain-Specific Optimizations With Fast Differentiation"

## Introduction

## Hardware Dependencies

| Component   | Minimum                 | Recommended                                                |
| ----------- | ----------------------- | ---------------------------------------------------------- |
| **CPU**     | 6 cores                 | 32 cores                                                   |
| **GPU**     | NVIDIA GPU (CUDA 12.9+) | NVIDIA GPU with 12 GiB+ VRAM (Blackwell, CUDA 12.9+)       |
| **Storage** | 10GB                    | 10GB                                                       |

We recommend a high CPU core count due to the overhead of building LLVM.

The evaluation in the paper was performed on an Ubuntu 24.04.4 LTS workstation with an NVIDIA GeForce RTX 5090 GPU with 32 GiB VRAM.

## Getting Started

We assume the target machine as a valid CUDA and GPU driver installation with CUDA 12.9+.

0. **Install NVIDIA container toolkit**

The following commands may be used to install the container toolkit on Ubuntu.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1
sudo apt-get install -y \
   nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For more details and instructions for other Linux distros, see [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. **Pull and build the docker container**

2. **Verify the installation**

## Step by Step Instructions

## Reusability Guide
