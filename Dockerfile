FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ARG USERNAME=evaluator

ENV NJOBS=100
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/$USERNAME
ENV WORKDIR=$HOME

WORKDIR $HOME

RUN groupadd -r $USERNAME && \
  useradd -r -g $USERNAME -d /home/$USERNAME -m $USERNAME

RUN apt-get -q update \
  && apt-get install -y --no-install-recommends ca-certificates software-properties-common curl gnupg2 \
    cmake gcc g++ clang-18 lld python3-venv python3-pip ninja-build git time sudo \
    opencl-headers

RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

RUN chown $USERNAME $HOME
USER $USERNAME

# put user local bin on PATH
RUN mkdir -p $HOME/.local/bin
ENV PATH="$HOME/.local/bin:$PATH"

# Install Bazelisk
RUN curl -fLO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 .local/bin/bazel && \
    chmod +x .local/bin/bazel

# Build Clang and MLIR (for EnzymeLLVM)
RUN git clone https://github.com/llvm/llvm-project.git \
  && cd llvm-project \
  && git checkout 9d618fc

RUN mkdir llvm-project/build && cd llvm-project/build \
  && cmake ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DLLVM_ENABLE_PROJECTS='clang;mlir' -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" -DLLVM_ENABLE_LLD=ON \
  && ninja -j $NJOBS \
  && cd ../..

# Build Enzyme
RUN git clone https://github.com/EnzymeAD/Enzyme.git \
  && cd Enzyme \
  && git checkout 4433004

RUN mkdir Enzyme/build && cd Enzyme/build \
  && cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  && ninja -j $NJOBS \
  && cd ../..

# Build Reactant (this is the EnzymeMLIR C++ frontend)
RUN git clone https://github.com/EnzymeAD/Reactant.git \
  && cd Reactant \
  && git checkout a38313b

RUN mkdir Reactant/build && cd Reactant/build \
  && cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  && ninja -j $NJOBS \
  && cd ../..

# Build Enzyme-JAX (This builds EnzymeMLIR with a plugins for both Reactant and JAX)
RUN git clone https://github.com/EnzymeAD/Enzyme-JAX \
  && cd Enzyme-JAX \
  && git checkout 5ff3f33
RUN cd Enzyme-JAX && CC=clang-18 HERMETIC_PYTHON_VERSION=3.12 bazel build :wheel :libRaise.so --jobs=$NJOBS

# Activate the Python virtual environment
RUN python3 -m venv eval-env
ENV PATH="$HOME/eval-env/bin:$PATH"

# TODO move environment to a `requirements.txt` file (use pip freeze)
# RUN pip install numpy==1.26.4 scipy==1.14.1 pandas==2.2.2 matplotlib==3.9.1.post1 "jax[cuda12]"
RUN pip install -r requirements.txt

COPY . $HOME
RUN sudo chown -R $USERNAME cpp
RUN sudo chown -R $USERNAME jax
