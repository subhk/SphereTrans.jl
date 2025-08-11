#!/usr/bin/env julia

# Simple test of SHTns_jll using a minimal approach similar to SHTns.jl/SHTnsSpheres.jl
# This test bypasses SHTnsKit.jl's complex fallback logic

println("Testing basic SHTns_jll functionality...")

try
    # Import SHTns_jll directly
    import SHTns_jll
    
    println("✅ SHTns_jll loaded successfully")
    println("Library path: $(SHTns_jll.LibSHTns)")
    
    # Test basic symbol availability
    using Libdl
    handle = dlopen(SHTns_jll.LibSHTns, RTLD_LAZY)
    
    # Check for basic symbols
    symbols = [:shtns_create, :shtns_set_grid, :shtns_sh_to_spat, :shtns_spat_to_sh, :shtns_free]
    
    println("\nSymbol availability:")
    for sym in symbols
        available = dlsym_e(handle, sym) != C_NULL
        println("  $sym: $(available ? "✅" : "❌")")
    end
    
    dlclose(handle)
    
    # Try a very basic SHTns setup similar to other packages
    println("\n🧪 Testing basic SHTns operations...")
    
    # Parameters similar to what other packages use
    lmax = 8
    mmax = 8  
    mres = 1
    
    # Create config
    cfg_ptr = ccall((:shtns_create, SHTns_jll.LibSHTns), Ptr{Cvoid},
                    (Cint, Cint, Cint), lmax, mmax, mres)
    
    if cfg_ptr == C_NULL
        println("❌ shtns_create failed")
        exit(1)
    end
    
    println("✅ shtns_create succeeded")
    
    # Set grid with very simple parameters
    nlat = 16  # Should be >= lmax + 1
    nphi = 32  # Should be >= 2*mmax + 1  
    grid_type = 0  # SHT_GAUSS
    
    println("Attempting shtns_set_grid with nlat=$nlat, nphi=$nphi, grid_type=$grid_type")
    
    # This is the call that should trigger "nlat or nphi is zero!" if there's an issue
    ccall((:shtns_set_grid, SHTns_jll.LibSHTns), Cvoid,
          (Ptr{Cvoid}, Cint, Cint, Cint), cfg_ptr, nlat, nphi, grid_type)
    
    println("✅ shtns_set_grid succeeded - no 'nlat or nphi is zero!' error")
    
    # Clean up
    ccall((:shtns_free, SHTns_jll.LibSHTns), Cvoid, (Ptr{Cvoid},), cfg_ptr)
    
    println("✅ Test completed successfully!")
    println("\n🎯 SHTns_jll appears to work fine with simple approach")
    println("The issue may be in SHTnsKit.jl's complex parameter handling")

catch e
    println("❌ Test failed: $e")
    
    if occursin("nlat or nphi is zero", string(e))
        println("\n🔍 Analysis: Got the 'nlat or nphi is zero!' error")
        println("This suggests the issue is with how parameters are passed to SHTns")
    elseif occursin("undefined symbol", string(e)) || occursin("symbol not found", string(e))
        println("\n🔍 Analysis: Missing symbols in SHTns_jll binary")
        println("The binary distribution may be incomplete")
    else
        println("\n🔍 Analysis: Other error - may be Julia/C interface issue")
    end
    
    exit(1)
end