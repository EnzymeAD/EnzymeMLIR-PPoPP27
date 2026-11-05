FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ARG USERNAME=evaluator
ARG NJOBS
ARG JULIA_VERSION=1.12.7

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/$USERNAME
ENV WORKDIR=$HOME

WORKDIR $HOME

RUN groupadd -r $USERNAME && \
  useradd -r -g $USERNAME -d /home/$USERNAME -m $USERNAME

RUN apt-get -q update \
  && apt-get install -y --no-install-recommends ca-certificates software-properties-common curl gnupg2 \
    cmake gcc g++ clang-18 lld python3-venv python3-pip ninja-build git time sudo \
    opencl-headers numactl procps linux-perf \
  && rm -rf /var/lib/apt/lists/*

RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Julia directly
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then JULIA_ARCH="x64"; else JULIA_ARCH="$ARCH"; fi && \
    curl -fL "https://julialang-s3.julialang.org/bin/linux/${JULIA_ARCH}/1.12/julia-${JULIA_VERSION}-linux-${ARCH}.tar.gz" | tar -xz -C /usr/local --strip-components=1

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
  && ninja -j ${NJOBS:-$(nproc)} \
  && cd ../..

# Build Enzyme
RUN git clone https://github.com/EnzymeAD/Enzyme.git \
  && cd Enzyme \
  && git checkout 4433004

RUN mkdir Enzyme/build && cd Enzyme/build \
  && cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  && ninja -j ${NJOBS:-$(nproc)} \
  && cd ../..

# Build Reactant (this is the EnzymeMLIR C++ frontend)
RUN git clone https://github.com/EnzymeAD/Reactant.git \
  && cd Reactant \
  && git checkout a38313b

RUN mkdir Reactant/build && cd Reactant/build \
  && cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  && ninja -j ${NJOBS:-$(nproc)} \
  && cd ../..

# Build Enzyme-JAX (This builds EnzymeMLIR with a plugins for both Reactant and JAX)
RUN git clone https://github.com/EnzymeAD/Enzyme-JAX \
  && cd Enzyme-JAX \
  && git checkout 5ff3f33
RUN cd Enzyme-JAX && CC=clang-18 HERMETIC_PYTHON_VERSION=3.12 bazel build :wheel :libRaise.so --jobs=${NJOBS:-$(nproc)}

# Activate the Python virtual environment
RUN python3 -m venv eval-env
ENV PATH="$HOME/eval-env/bin:$PATH"

# Install Python dependencies
COPY --chown=$USERNAME:$USERNAME requirements.txt $HOME/requirements.txt
RUN pip install -r requirements.txt
RUN pip install Enzyme-JAX/bazel-bin/enzyme_ad-0.0.10-py312-none-manylinux2014_x86_64.whl

# Configure Julia and precompile relevant package versions (kept in the docker dir)
ENV JULIA_DEPOT_PATH=/opt/julia_depot \
    JULIA_NUM_THREADS=1 \
    JULIA_PKG_PRECOMPILE_AUTO=0

USER root
RUN mkdir -p /opt/julia_depot /opt/julia_project \
 && chown -R $USERNAME:$USERNAME /opt/julia_depot /opt/julia_project
USER $USERNAME

COPY --chown=$USERNAME:$USERNAME docker/Project.toml /opt/julia_project/Project.toml

RUN if [ "$(uname -m)" = "aarch64" ]; then \
        echo 'generic;neoverse-v2,clone_all' > /opt/julia_cpu_target; \
    else \
        echo 'generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)' > /opt/julia_cpu_target; \
    fi \
 && export JULIA_CPU_TARGET="$(cat /opt/julia_cpu_target)" \
 && julia --project=/opt/julia_project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.status()'

USER root
RUN chmod -R a+rX /opt/julia_depot /opt/julia_project
USER $USERNAME

# Stack a writable /tmp depot in front of the precompiled /opt depot at runtime
ENV JULIA_DEPOT_PATH=/tmp/julia_depot:/opt/julia_depot \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"

COPY --chown=$USERNAME:$USERNAME . $HOME
