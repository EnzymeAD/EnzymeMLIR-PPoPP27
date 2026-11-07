FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ARG USERNAME=evaluator
ARG NJOBS

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/$USERNAME
ENV WORKDIR=$HOME

WORKDIR $HOME

RUN groupadd -r $USERNAME && \
  useradd -r -g $USERNAME -d /home/$USERNAME -m $USERNAME

RUN apt-get -q update \
  && apt-get install -y --no-install-recommends ca-certificates software-properties-common curl gnupg2 \
    cmake gcc g++ clang-18 lld python3-venv python3-pip ninja-build git time sudo \
    opencl-headers \
    cuda-nsight-systems-12-9 \
  && rm -rf /var/lib/apt/lists/*

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
RUN git init -q llvm-project \
  && cd llvm-project \
  && git fetch -q --depth 1 https://github.com/llvm/llvm-project.git 9d618fc0243103263ebf9bdcd08ef25e871d9c78 \
  && git checkout -q FETCH_HEAD \
  && rm -rf .git

RUN mkdir llvm-project/build && cd llvm-project/build \
  && cmake ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DLLVM_ENABLE_PROJECTS='clang;mlir' -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" -DLLVM_ENABLE_LLD=ON \
  && ninja -j ${NJOBS:-$(nproc)} \
  && find . -name '*.o' -delete

# Build Enzyme
RUN git init -q Enzyme \
  && cd Enzyme \
  && git fetch -q --depth 1 https://github.com/EnzymeAD/Enzyme.git 4433004499bb2a38076f5a4608ffa3012efc19a5 \
  && git checkout -q FETCH_HEAD \
  && rm -rf .git

RUN mkdir Enzyme/build && cd Enzyme/build \
  && cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  && ninja -j ${NJOBS:-$(nproc)} \
  && find . -name '*.o' -delete

# Build Reactant (this is the EnzymeMLIR C++ frontend)
RUN git init -q Reactant \
  && cd Reactant \
  && git fetch -q --depth 1 https://github.com/EnzymeAD/Reactant.git a38313b8decc4861eb127a57bab5275eb08968b9 \
  && git checkout -q FETCH_HEAD \
  && rm -rf .git

RUN mkdir Reactant/build && cd Reactant/build \
  && cmake ../enzyme -G Ninja -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  && ninja -j ${NJOBS:-$(nproc)} \
  && find . -name '*.o' -delete

# Build libRaise.so (EnzymeMLIR raising plugin for Reactant C++ frontend) and discard Bazel cache
RUN git init -q Enzyme-JAX \
  && cd Enzyme-JAX \
  && git fetch -q --depth 1 https://github.com/EnzymeAD/Enzyme-JAX.git 5ff3f3308af08ab9e6cfb19d77e4867ce4afd271 \
  && git checkout -q FETCH_HEAD \
  && CC=clang-18 HERMETIC_PYTHON_VERSION=3.12 bazel build :libRaise.so --jobs=${NJOBS:-$(nproc)} \
  && cp -L bazel-bin/libRaise.so /tmp/libRaise.so \
  && bazel clean --expunge \
  && cd .. \
  && rm -rf Enzyme-JAX $HOME/.cache/bazel \
  && mkdir -p Enzyme-JAX/bazel-bin \
  && mv /tmp/libRaise.so Enzyme-JAX/bazel-bin/libRaise.so \
  && chmod 0755 Enzyme-JAX/bazel-bin/libRaise.so

# Activate the Python virtual environment
RUN python3 -m venv eval-env
ENV PATH="$HOME/eval-env/bin:$PATH"

# Install Python dependencies and latest Enzyme-JAX (enzyme-ad) from PyPI
COPY --chown=$USERNAME:$USERNAME docker/requirements.txt $HOME/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir --ignore-requires-python --upgrade enzyme-ad

# Install system utilities needed for Julia benchmarks (numactl, procps) and download Julia directly
USER root
RUN apt-get -q update \
  && apt-get install -y --no-install-recommends numactl procps \
  && rm -rf /var/lib/apt/lists/*

ARG JULIA_VERSION=1.12.7
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then JULIA_ARCH="x64"; else JULIA_ARCH="$ARCH"; fi && \
    curl -fL "https://julialang-s3.julialang.org/bin/linux/${JULIA_ARCH}/1.12/julia-${JULIA_VERSION}-linux-${ARCH}.tar.gz" | tar -xz -C /usr/local --strip-components=1

# Configure Julia and precompile relevant package versions (kept in the docker dir)
ENV JULIA_DEPOT_PATH=/opt/julia_depot \
    JULIA_NUM_THREADS=1 \
    JULIA_PKG_PRECOMPILE_AUTO=0

RUN mkdir -p /opt/julia_depot /opt/julia_project \
 && chown -R $USERNAME:$USERNAME /opt/julia_depot /opt/julia_project
USER $USERNAME

COPY --chown=$USERNAME:$USERNAME docker/Project.toml /opt/julia_project/Project.toml

RUN if [ "$(uname -m)" = "aarch64" ]; then \
        TARGET="generic;neoverse-v2,clone_all"; \
    else \
        TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)"; \
    fi \
 && export JULIA_CPU_TARGET="$TARGET" \
 && julia --project=/opt/julia_project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.status()' \
 && chmod -R a+rX /opt/julia_depot /opt/julia_project

USER root
RUN mkdir -p /github \
 && chown $USERNAME:$USERNAME /github \
 && chmod 0777 /github

# Stack a writable /tmp depot in front of the precompiled /opt depot at runtime
ENV JULIA_DEPOT_PATH=/tmp/julia_depot:/opt/julia_depot \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"

COPY --chown=$USERNAME:$USERNAME . $HOME
