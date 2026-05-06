#!/usr/bin/env bash
# build.sh — one-shot builder for QuEST + QC_test_GHZ.
#
# Usage:  ./build.sh <mode>
#   cpu        single-node, CPU + OpenMP                  (smoke test only)
#   cpu-mpi    multi-node, CPU + OpenMP + MPI
#   gpu-cuda   multi-node, NVIDIA GPU (CUDA) + MPI
#   gpu-hip    multi-node, AMD GPU (HIP)   + MPI
#
# Optional environment variables (override compilers/architectures):
#   CC, CXX            user-side C / C++ compilers     (default: gcc/mpicc)
#   QUEST_CC, QUEST_CXX  compilers QuEST is configured with
#   CUDA_ARCH          CUDA compute capability         (e.g. 80, 90)
#   HIP_ARCH           HIP gfx code                    (e.g. gfx90a, gfx940)
#   CXX_RUNTIME        C++ runtime link flag           (default: auto-detect)
#
# Re-running with a different <mode> automatically rebuilds QuEST from scratch.
# Re-running with the same <mode> only rebuilds QC_test_GHZ.c (incremental).

set -euo pipefail

mode="${1:-}"
case "$mode" in
    cpu)        QFLAGS="-DENABLE_MULTITHREADING=ON"
                : "${CC:=gcc}"; USE_MPI="" ;;
    cpu-mpi)    QFLAGS="-DENABLE_MULTITHREADING=ON -DENABLE_DISTRIBUTION=ON"
                : "${CC:=mpicc}"; USE_MPI="-DUSE_MPI" ;;
    gpu-cuda)   QFLAGS="-DENABLE_MULTITHREADING=ON -DENABLE_DISTRIBUTION=ON -DENABLE_CUDA=ON"
                [ -n "${CUDA_ARCH:-}" ] && QFLAGS+=" -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
                : "${CC:=mpicc}"; USE_MPI="-DUSE_MPI" ;;
    gpu-hip)    QFLAGS="-DENABLE_MULTITHREADING=ON -DENABLE_DISTRIBUTION=ON -DENABLE_HIP=ON"
                [ -n "${HIP_ARCH:-}"  ] && QFLAGS+=" -DCMAKE_HIP_ARCHITECTURES=$HIP_ARCH"
                : "${CC:=mpicc}"; USE_MPI="-DUSE_MPI" ;;
    *) sed -n '3,17p' "$0"; exit 1 ;;
esac

Q="QuEST"
B="$Q/build"
QUEST_REPO="${QUEST_REPO:-https://github.com/QuEST-Kit/QuEST.git}"
QUEST_TAG="${QUEST_TAG:-v4.2.0}"

# Step 0: fetch QuEST if missing (pinned to a known-good tag for reproducibility)
if [ ! -d "$Q/.git" ]; then
    echo "==> Cloning QuEST $QUEST_TAG from $QUEST_REPO"
    rm -rf "$Q"
    git clone --depth 1 --branch "$QUEST_TAG" "$QUEST_REPO" "$Q"
else
    have="$(cd "$Q" && git describe --tags --always 2>/dev/null || echo unknown)"
    if [ "$have" != "$QUEST_TAG" ]; then
        echo "WARNING: QuEST/ is at '$have', expected '$QUEST_TAG'."
        echo "         Re-run with QUEST_TAG=$have ./build.sh ... to suppress, or"
        echo "         rm -rf QuEST  to fetch the pinned version."
    fi
fi

# macOS: libomp must be discoverable; auto-set OpenMP_ROOT if brew has it.
if [ "$(uname -s)" = "Darwin" ] && [ -z "${OpenMP_ROOT:-}" ] && command -v brew >/dev/null; then
    if [ -d "$(brew --prefix)/opt/libomp" ]; then
        export OpenMP_ROOT="$(brew --prefix)/opt/libomp"
    fi
fi

# Step 1: rebuild QuEST if mode changed
if [ ! -f "$B/.mode" ] || [ "$(cat "$B/.mode" 2>/dev/null)" != "$mode" ]; then
    echo "==> Building QuEST ($mode)"
    rm -rf "$B"
    cmake -B "$B" -S "$Q" $QFLAGS \
        ${QUEST_CC:+-DCMAKE_C_COMPILER="$QUEST_CC"} \
        ${QUEST_CXX:+-DCMAKE_CXX_COMPILER="$QUEST_CXX"} \
        -DCMAKE_BUILD_TYPE=Release \
        -Wno-dev
    cmake --build "$B" --parallel
    echo "$mode" > "$B/.mode"
else
    echo "==> QuEST already built for mode='$mode' (skipping)"
fi

# Step 2: pick C++ runtime needed to link against QuEST core (which is C++)
if [ -z "${CXX_RUNTIME:-}" ]; then
    case "$(uname -s)" in
        Darwin) CXX_RUNTIME="-lc++" ;;
        *)      CXX_RUNTIME="-lstdc++" ;;
    esac
fi

# Step 3: build our user binary
echo "==> Building QC_test_GHZ ($mode, $CC)"
RPATH="$(cd "$B" && pwd)"
$CC -O3 -std=gnu11 $USE_MPI \
    -I"$Q/quest/include" -I"$Q" -I"$B" \
    QC_test_GHZ.c \
    -L"$B" -lQuEST -Wl,-rpath,"$RPATH" $CXX_RUNTIME -lm \
    -o QC_test_GHZ

echo "==> OK   binary: $PWD/QC_test_GHZ   mode: $mode"
