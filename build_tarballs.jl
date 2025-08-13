using BinaryBuilder, Pkg

name = "SHTnsKit"
version = v"1.0.0"

# Since we have a pre-built library, we'll use it directly
sources = [
    DirectorySource("./")  # Current directory contains libshtns_omp.so
]

# Script to install our pre-built library
script = raw"""
cd ${WORKSPACE}/srcdir

# Create the lib directory
mkdir -p ${libdir}

# Copy our pre-built library to the appropriate location
if [[ ${target} == x86_64-linux-gnu ]]; then
    # For Linux x86_64, copy the existing .so file
    if [[ -f libshtns_omp.so ]]; then
        cp libshtns_omp.so ${libdir}/libshtns.so
        # Also create the OMP version
        cp libshtns_omp.so ${libdir}/libshtns_omp.so
        # Set proper permissions
        chmod 755 ${libdir}/libshtns.so
        chmod 755 ${libdir}/libshtns_omp.so
        echo "Installed libshtns for Linux x86_64"
    else
        echo "ERROR: libshtns_omp.so not found!"
        exit 1
    fi
elif [[ ${target} == x86_64-apple-darwin* ]]; then
    # For macOS, we'll need a .dylib version
    # This would require recompilation for macOS
    echo "WARNING: macOS build requires recompilation from source"
    # You would need to build from SHTns source for macOS
    touch ${libdir}/libshtns.dylib
elif [[ ${target} == x86_64-w64-mingw32 ]]; then
    # For Windows, we'll need a .dll version  
    # This would require recompilation for Windows
    echo "WARNING: Windows build requires recompilation from source"
    # You would need to build from SHTns source for Windows
    touch ${libdir}/libshtns.dll
fi
"""

# For now, let's focus on Linux x86_64 where we have the actual library
platforms = [
    Platform("x86_64", "linux"; libc="glibc")
]

# Products - what the package will export
products = [
    LibraryProduct("libshtns", :libshtns),
    LibraryProduct("libshtns_omp", :libshtns_omp)  # OpenMP version
]

# Dependencies
dependencies = [
    # Add any runtime dependencies here if needed
    # For example:
    # Dependency("OpenMP_jll"; compat="4.0.1")
    # Dependency("FFTW_jll")
]

# Build the tarballs
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6",
               preferred_gcc_version=v"8")