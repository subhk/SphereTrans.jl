using BinaryBuilder

name = "SHTnsKit"
version = v"1.0.0"

# Source of SHTns (pin to a known-good commit or tag)
sources = [
    GitSource("https://bitbucket.org/nschaeff/shtns.git", "b3b0ec4cb7d93b9b5a70e3e2e9f1ab1e80e9b5a3"),
]

script = raw"""
cd ${WORKSPACE}/srcdir/shtns*

# Avoid -march=native for cross-compilation
export CFLAGS="-O3"
export CXXFLAGS="-O3"

# OpenMP flags vary by platform/compilers in BB
if [[ ${target} == *-apple-darwin* ]]; then
    # Apple: link against libomp from OpenMP_jll
    export CFLAGS="${CFLAGS} -Xpreprocessor -fopenmp"
    export CXXFLAGS="${CXXFLAGS} -Xpreprocessor -fopenmp"
    export LDFLAGS="${LDFLAGS} -lomp"
elif [[ ${target} == *-mingw* ]]; then
    # MinGW (Windows): -fopenmp is provided by compilers
    export CFLAGS="${CFLAGS} -fopenmp"
    export CXXFLAGS="${CXXFLAGS} -fopenmp"
    export LDFLAGS="${LDFLAGS} -fopenmp"
else
    # Linux and others
    export CFLAGS="${CFLAGS} -fopenmp"
    export CXXFLAGS="${CXXFLAGS} -fopenmp"
    export LDFLAGS="${LDFLAGS} -fopenmp"
fi

# Configure: use FFTW_jll (installed into ${prefix} by BB)
./configure \
    --prefix=${prefix} \
    --host=${target} \
    --enable-openmp \
    --enable-shared --disable-static \
    --with-fftw3=${prefix}

make -j${nproc}
make install

# List outputs for debugging
ls -la ${libdir}
"""

platforms = expand_cxxstring_abis(vcat(
    Platform("x86_64", "linux"; libc="glibc"),
    Platform("aarch64", "linux"; libc="glibc"),
    Platform("x86_64", "macos"),
    Platform("aarch64", "macos"),
    Platform("x86_64", "windows"),
))

products = [
    LibraryProduct("libshtns", :libshtns),
]

dependencies = [
    Dependency("FFTW_jll"),
    Dependency("OpenMP_jll"),
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6")

