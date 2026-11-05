#===============================================================================
# User Options
#===============================================================================

CUDA_PATH ?= /usr/local/cuda-12.9

EMBEDDED_CLANG ?= no
USE_ENZYME_LLVM ?= no

CLANG_ENZYME ?= $(HOME)/Enzyme/build/Enzyme/ClangEnzyme-24.so
REACTANT_PATH ?= $(HOME)/Reactant/build/Enzyme/libClangReactant-24.so
CLANG_PATH ?= $(HOME)/llvm-project/build/bin/clang++
LIB_RAISE_PATH ?= $(HOME)/Enzyme-JAX/bazel-bin/libRaise.so
REACTANT_EXTRA_PATH ?= $(HOME)/git/Reactant.jl/deps/ReactantExtra/bazel-bin/

export REACTANT_DEBUG=0
export PATH := $(CUDA_PATH)/bin:$(PATH)
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE = 128
export LD_LIBRARY_PATH := $(REACTANT_EXTRA_PATH):$(LD_LIBRARY_PATH)

OPTIMIZE    ?= yes

NEWCACHE    ?= yes
AA          ?= no
PHISTRUCT   ?= no
FORWARD     ?= no
COALESE     ?= yes
CACHELICM   ?= yes
ABI         ?= yes
ALLOCATOR   ?= no
READONLY    ?= yes
SPLITMULTI  ?= yes
AFFINEAD    ?= yes
RMATOMICS   ?= yes
SORTBLOCK   ?= yes
HOISTALLOC  ?= yes
VERIFY      ?= no

# Binomial checkpointing of the outer kernel loop (requires FORWARD=no)
BINOMIAL        ?= no
BINOMIAL_BUDGET ?= 4
MINCUT          ?= yes

DEBUG       = no
PROFILE     = no
SM_VERSION  ?= 120

#===============================================================================
# Program name & source code list
#===============================================================================

program = lbm

source = args.cc lbm.cu parboil_cuda.cc main.cc

obj = $(patsubst %.cc,%.o,$(patsubst %.cu,%.o,$(source)))

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -mllvm -max-heap-to-stack-size=1000000 -I $(CUDA_PATH)/include -I .
# -mllvm -max-heap-to-stack-size=-1
# -Rpass=attributor -mllvm -debug -mllvm -debug-only=attributor


CC := $(CLANG_PATH)

ENZYME_LOAD_FLAGS :=
ifeq ($(USE_ENZYME_LLVM),yes)
  ENZYME_LOAD_FLAGS += -fplugin=$(CLANG_ENZYME)
else
  ifeq ($(EMBEDDED_CLANG),no)
    ENZYME_LOAD_FLAGS += -fplugin=$(REACTANT_PATH) -mllvm -raising-plugin-path -mllvm $(LIB_RAISE_PATH)
  endif
  CFLAGS += -mllvm -reactant-backend -mllvm cuda -mllvm -reactant-dataflow

  ifeq ($(READONLY),yes)
    # Avoid caching loads of pointers that are never overwritten
    CFLAGS += -mllvm -reactant-mark-readonly
  endif

  ifeq ($(AFFINEAD),yes)
    CFLAGS += -mllvm -reactant-pre-ad-lower-affine=0
  else
    # Lower affine dialect operations before AD (https://github.com/EnzymeAD/Enzyme/issues/3031)
    CFLAGS += -mllvm -reactant-pre-ad-lower-affine=1
  endif

  ifeq ($(SPLITMULTI),yes)
    # Split certain multi-result ops into separate ops. Can lead to better mincut solutions
    CFLAGS += -mllvm -reactant-split-multi-results
  endif

  ifeq ($(RMATOMICS),yes)
    # Turn the atomics the reverse pass introduces back into load/store where
    # polyhedral analysis proves no race exists
    CFLAGS += -mllvm -reactant-remove-atomics=1
  else
    CFLAGS += -mllvm -reactant-remove-atomics=0
  endif

  ifeq ($(SORTBLOCK),yes)
    # Cluster the resulting loads at the top of their block so they overlap
    # instead of forming a chain of dependent round trips
    CFLAGS += -mllvm -reactant-sort-block-memory=1
  else
    CFLAGS += -mllvm -reactant-sort-block-memory=0
  endif

  ifeq ($(HOISTALLOC),yes)
    # Move the allocations the reverse pass makes per iteration out of the loop
    CFLAGS += -mllvm -reactant-hoist-loop-allocations=1
  else
    CFLAGS += -mllvm -reactant-hoist-loop-allocations=0
  endif

  # Only the Reactant plugin understands enzyme_set_mincut, so ENZYME_MINCUT is
  # defined exactly when the feature is available (see the guard in lbm.cu)
  ifeq ($(MINCUT),yes)
    CFLAGS += -DENZYME_MINCUT=1
  else
    CFLAGS += -DENZYME_MINCUT=0
  endif
endif

CFLAGS += -ffast-math --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 --cuda-gpu-arch=sm_$(SM_VERSION)  -std=c++11 $(ENZYME_LOAD_FLAGS) -fno-exceptions
  

# Linker Flags
LDFLAGS = -lcudart_static "-ldl" "-lrt" -lpthread -lm -L$(REACTANT_EXTRA_PATH) # -lReactantExtra

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(ALLOCATOR),yes)
  CFLAGS += -DALLOCATOR
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif

ifeq ($(ABI),yes)
  CFLAGS += -DABI
  CFLAGS += -DSIZE=20
else
  CFLAGS += -DSIZE=8
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


# Optimization Flags
ifeq ($(FORWARD),yes)
  CFLAGS += 
else
  CFLAGS += -DALLOW_AD=1
  ifeq ($(USE_ENZYME_LLVM),yes)
    CFLAGS += -DENZYME_LLVM_AD=1
  endif
endif

# Binomial checkpointing Flags
ifeq ($(BINOMIAL),yes)
  CFLAGS += -DBINOMIAL_CHECKPOINTING=1 -DBINOMIAL_BUDGET=$(BINOMIAL_BUDGET)
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) lbm_kernel.cu Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu lbm.h main.h parboil.h lbm_macros.h lbm_kernel.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cc lbm.h main.h parboil.h lbm_macros.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) *.o *.re_export

# Time every AD configuration for each step count in RUN_STEPS, writing
# timings/summary_<steps>.txt (plotted by lbm.jl). The binaries are built on
# the first step count only, pass SKIP_BUILD=1 to reuse existing ones.
# Other knobs of run_timings.sh (REPS, WARMUP, MEMORY, ...) can be set here too.
# The toolchain paths default to the ones above, i.e. the Docker image layout;
# outside of it pass e.g. REACTANT_BIN=... ENZYME_ROOT=... (see run_timings.sh).
RUN_STEPS ?= 25 50 100 200 400 800 1200

run:
	@skip=$${SKIP_BUILD:-0}; for n in $(RUN_STEPS); do \
	  SKIP_BUILD=$$skip ./run_timings.sh $$n || exit 1; skip=1; \
	done

.PHONY: run clean
