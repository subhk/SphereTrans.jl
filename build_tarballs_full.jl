using BinaryBuilder, Pkg

name = "SHTnsKit"
version = v"1.0.0"

# Sources - SHTns library source code
sources = [
    GitSource("https://bitbucket.org/nschaeff/shtns.git", "b3b0ec4cb7d93b9b5a70e3e2e9f1ab1e80e9b5a3"),  # Latest stable commit
    # Also include your local optimized library as reference
    DirectorySource("./")
]

# Build script for SHTns with OpenMP support
script = raw"""
cd ${WORKSPACE}/srcdir/shtns*

# Configure build with OpenMP and optimization flags
export CFLAGS="-O3 -march=native -fopenmp"
export CXXFLAGS="-O3 -march=native -fopenmp"
export LDFLAGS="-fopenmp"

if [[ ${target} == *-mingw* ]]; then
    # Windows-specific configuration
    ./configure --prefix=${prefix} --host=${target} \
                --enable-openmp \
                --enable-static --enable-shared \
                --with-fftw3=${prefix} \
                CFLAGS="-O3 -fopenmp" \
                LDFLAGS="-fopenmp -L${libdir}" \
                CC=${CC} CXX=${CXX}
elif [[ ${target} == *-apple-darwin* ]]; then
    # macOS-specific configuration
    ./configure --prefix=${prefix} --host=${target} \
                --enable-openmp \
                --enable-static --enable-shared \
                --with-fftw3=${prefix} \
                CFLAGS="-O3 -Xpreprocessor -fopenmp" \
                CXXFLAGS="-O3 -Xpreprocessor -fopenmp" \
                LDFLAGS="-lomp -L${libdir}" \
                CC=${CC} CXX=${CXX}
else
    # Linux and other Unix-like systems
    ./configure --prefix=${prefix} --host=${target} \
                --enable-openmp \
                --enable-static --enable-shared \
                --with-fftw3=${prefix} \
                CFLAGS="-O3 -fopenmp" \
                CXXFLAGS="-O3 -fopenmp" \
                LDFLAGS="-fopenmp -L${libdir}" \
                CC=${CC} CXX=${CXX}
fi

# Build and install
make -j${nproc}
make install

# For Linux x86_64, also copy our pre-optimized version if available
if [[ ${target} == x86_64-linux-gnu ]]; then
    cd ${WORKSPACE}/srcdir
    if [[ -f libshtns_omp.so ]]; then
        echo "Installing pre-optimized library for Linux x86_64"
        cp libshtns_omp.so ${libdir}/libshtns_optimized.so
        chmod 755 ${libdir}/libshtns_optimized.so
    fi
fi

# Verify the library was built correctly
ls -la ${libdir}/libshtns*
"""

# Platforms to build for
platforms = [
    Platform("x86_64", "linux"; libc="glibc"),
    Platform("x86_64", "macos"),
    Platform("aarch64", "macos"),
    Platform("x86_64", "windows"),
    Platform("aarch64", "linux"; libc="glibc")
]

# Products - libraries to export
products = [
    LibraryProduct("libshtns", :libshtns),
    # Include the optimized version for Linux x86_64
    LibraryProduct("libshtns_optimized", :libshtns_optimized; platforms=[Platform("x86_64", "linux"; libc="glibc")])
]

# Dependencies
dependencies = [
    Dependency("FFTW_jll"),
    Dependency("OpenMP_jll"; compat="4.0.1")
]

# Build the tarballs
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6",
               preferred_gcc_version=v"8")