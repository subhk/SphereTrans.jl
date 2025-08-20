# NOTE: Fill in the `sources` URL and `sha256` below to match the SHTns release you want.
# Then run this script with BinaryBuilder to produce platform tarballs and (optionally) a JLL.

using BinaryBuilder

name = "SHTnsKit"
version = v"1.0.0"  # library version, not the JLL build number

# Upstream source of SHTns (placeholder; update to the exact URL + sha256)
sources = [
    ArchiveSource(
        # Example: official SHTns source release tarball
        # "https://github.com/SHTns/shtns/archive/refs/tags/v3.7.1.tar.gz",
        # "<sha256-of-source-tarball>"
        "https://example.com/shtns-vX.Y.Z.tar.gz",
        "<sha256>"
    ),
]

# Define the platforms we want to build for
platforms = [
    Platform("x86_64", "linux"; libc = "glibc"),
    Platform("aarch64", "macos"),
    Platform("x86_64", "macos"),
]

# Optional: Add platform-specific dependencies. macOS needs LLVMOpenMP for -fopenmp.
dependencies = Dependency[
    Dependency("LLVMOpenMP_jll"; platforms = filter(Sys.isapple, platforms)),
]

products = Product[
    LibraryProduct("libshtns", :libshtns),
    # Some distributions also provide an OpenMP-suffixed library; mark as optional
    LibraryProduct("libshtns_omp", :libshtns_omp; dont_dlopen=true, optional=true),
]

script = raw"""
#!/bin/bash
set -euxo pipefail

# SHTns can be built via configure+make or via CMake depending on version.
# We'll try CMake first; if not present, fall back to autotools.

cd ${WORKSPACE}/srcdir
SRCDIR=$(find . -maxdepth 1 -type d -name "shtns*" | head -n1)
cd "$SRCDIR"

# Always use our toolchain env vars
export CC=${CC}
export CXX=${CXX}
export FC=${FC:-}
export CFLAGS="${CFLAGS}"
export CXXFLAGS="${CXXFLAGS}"
export LDFLAGS="${LDFLAGS}"

mkdir -p build && cd build

if [ -f ../CMakeLists.txt ]; then
    cmake -DCMAKE_INSTALL_PREFIX=${prefix} \
          -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
          -DBUILD_SHARED_LIBS=ON \
          -DSHTNS_USE_OPENMP=ON \
          ..
    cmake --build . --parallel ${nproc}
    cmake --install .
else
    cd ..
    if [ -x ./configure ]; then
        ./configure --prefix=${prefix} CFLAGS="${CFLAGS}" LDFLAGS="${LDFLAGS}" CC=${CC} CXX=${CXX}
        make -j${nproc}
        make install
    else
        # Fallback: simple Makefile with PREFIX support
        make -j${nproc} CC=${CC} CXX=${CXX} CFLAGS="${CFLAGS}" LDFLAGS="${LDFLAGS}" || true
        mkdir -p ${libdir}
        # Try to find produced shared libraries
        find . -name "*.so*" -o -name "*.dylib" -maxdepth 3 -print -exec cp -vf {} ${libdir}/ \;
        # If headers are present, stage them too
        if [ -d include ]; then
            mkdir -p ${includedir}
            cp -r include/* ${includedir}/ || true
        fi
    fi
fi

# Some builds create libshtns_omp separately; if both exist, keep both.
ls -la ${libdir} || true

"""

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               preferred_gcc_version=v"7")
