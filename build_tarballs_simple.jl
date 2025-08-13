using BinaryBuilder, Pkg

name = "SHTnsKit"
version = v"1.0.0"

# Use your existing library
sources = [
    DirectorySource("./")
]

# Simple script to package your existing library
script = raw"""
cd ${WORKSPACE}/srcdir

# Create lib directory
mkdir -p ${libdir}

# Install the library based on platform
if [[ ${target} == x86_64-linux-gnu ]]; then
    if [[ -f libshtns_omp.so ]]; then
        # Install as both regular and OMP versions
        cp libshtns_omp.so ${libdir}/libshtns.so
        cp libshtns_omp.so ${libdir}/libshtns_omp.so
        chmod 755 ${libdir}/libshtns.so
        chmod 755 ${libdir}/libshtns_omp.so
        echo "Successfully installed SHTns library for Linux x86_64"
        
        # Verify symbols
        if command -v nm >/dev/null 2>&1; then
            echo "Library symbols check:"
            nm -D ${libdir}/libshtns.so | grep shtns_create || echo "Warning: shtns_create not found"
        fi
    else
        echo "ERROR: libshtns_omp.so not found in source directory"
        exit 1
    fi
else
    echo "ERROR: Platform ${target} not supported in this simple build"
    echo "Use build_tarballs_full.jl to build from source for other platforms"
    exit 1
fi
"""

# Only build for Linux x86_64 since that's what we have
platforms = [
    Platform("x86_64", "linux"; libc="glibc")
]

# Products
products = [
    LibraryProduct("libshtns", :libshtns),
    LibraryProduct("libshtns_omp", :libshtns_omp)
]

# No dependencies for the simple version
dependencies = Dependency[]

# Build
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6")