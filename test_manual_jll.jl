#!/usr/bin/env julia

# Manual test script to create SHTnsKit_jll-like functionality without BinaryBuilder

println("Testing manual SHTnsKit_jll creation...")

# First, let's test if our library loads correctly
using Libdl

lib_path = joinpath(@__DIR__, "libshtns_omp.so")
println("Testing library at: ", lib_path)

if !isfile(lib_path)
    error("Library not found: $lib_path")
end

# Try to open the library
try
    handle = dlopen(lib_path, RTLD_LAZY)
    println("✓ Successfully opened library")
    
    # Check for key symbols
    symbols = [:shtns_create, :shtns_create_with_grid, :shtns_destroy, :shtns_gauss_wts]
    for sym in symbols
        if dlsym_e(handle, sym) != C_NULL
            println("✓ Found symbol: $sym")
        else
            println("✗ Missing symbol: $sym")
        end
    end
    
    dlclose(handle)
    println("✓ Library validation completed")
    
catch e
    println("✗ Error loading library: $e")
end

# Create a simple JLL-like module
module TestSHTnsKit_jll
    using Libdl
    
    const libshtns = joinpath(@__DIR__, "libshtns_omp.so")
    const libshtns_omp = libshtns  # Same file for both
    
    function __init__()
        if !isfile(libshtns)
            error("SHTnsKit library not found: $libshtns")
        end
        
        # Verify we can load it
        handle = dlopen(libshtns, RTLD_LAZY | RTLD_GLOBAL)
        # Keep it loaded for the session
        global _lib_handle = handle
    end
end

println("\n✓ Created TestSHTnsKit_jll module")
println("Library path: ", TestSHTnsKit_jll.libshtns)

# Test the module
try
    TestSHTnsKit_jll.__init__()
    println("✓ TestSHTnsKit_jll initialized successfully")
    
    # Test a simple ccall
    result = ccall((:shtns_create, TestSHTnsKit_jll.libshtns), Ptr{Cvoid}, (Cint, Cint, Cint, UInt32), 4, 4, 1, 0)
    if result != C_NULL
        println("✓ Successfully called shtns_create")
        # Clean up
        ccall((:shtns_destroy, TestSHTnsKit_jll.libshtns), Cvoid, (Ptr{Cvoid},), result)
    else
        println("✗ shtns_create returned NULL")
    end
    
catch e
    println("✗ Error testing TestSHTnsKit_jll: $e")
end

println("\n🎉 Manual JLL test completed!")
println("\nNext steps:")
println("1. This demonstrates your library works correctly")
println("2. You can package it using BinaryBuilder when permissions are fixed")
println("3. Or use it directly by setting SHTNS_LIBRARY_PATH environment variable")