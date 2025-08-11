#!/usr/bin/env julia

# Debug SHTns calls to understand the "nlat or nphi is zero!" error

println("🔍 Debugging SHTns_jll parameter passing...")

try
    import SHTns_jll
    using Libdl
    
    # Test different parameter combinations
    test_cases = [
        (lmax=2, mmax=2, mres=1, nlat=4, nphi=8),    # Minimal case  
        (lmax=4, mmax=4, mres=1, nlat=8, nphi=16),   # Small case
        (lmax=8, mmax=8, mres=1, nlat=16, nphi=32),  # Previous case
        (lmax=8, mmax=8, mres=1, nlat=17, nphi=33),  # Odd numbers
        (lmax=8, mmax=8, mres=1, nlat=20, nphi=40),  # Larger grid
    ]
    
    for (i, case) in enumerate(test_cases)
        println("\n--- Test Case $i ---")
        println("Parameters: $(case)")
        
        try
            # Create config
            cfg_ptr = ccall((:shtns_create, SHTns_jll.LibSHTns), Ptr{Cvoid},
                           (Cint, Cint, Cint), case.lmax, case.mmax, case.mres)
            
            if cfg_ptr == C_NULL
                println("❌ shtns_create failed")
                continue
            end
            
            println("✅ shtns_create succeeded (ptr=$cfg_ptr)")
            
            # Try to set grid
            ccall((:shtns_set_grid, SHTns_jll.LibSHTns), Cvoid,
                  (Ptr{Cvoid}, Cint, Cint, Cint), cfg_ptr, case.nlat, case.nphi, 0)
            
            println("✅ shtns_set_grid succeeded!")
            
            # Clean up
            ccall((:shtns_free, SHTns_jll.LibSHTns), Cvoid, (Ptr{Cvoid},), cfg_ptr)
            println("✅ Cleaned up successfully")
            
        catch e
            println("❌ Failed: $e")
            if occursin("nlat or nphi is zero", string(e))
                println("   This suggests parameters are being received as zero by SHTns")
                println("   Possible C ABI or calling convention issue")
            end
        end
    end
    
    # Test with direct inspection of parameters
    println("\n🔍 Testing parameter inspection...")
    
    # Check if the issue is in how Julia passes Cint parameters
    lmax, mmax, mres = 4, 4, 1
    nlat, nphi = 8, 16
    
    println("Julia values: lmax=$lmax, mmax=$mmax, mres=$mres, nlat=$nlat, nphi=$nphi")
    println("Cint values: $(Cint(lmax)), $(Cint(mmax)), $(Cint(mres)), $(Cint(nlat)), $(Cint(nphi))")
    
    # Try with explicit Cint conversion
    cfg_ptr = ccall((:shtns_create, SHTns_jll.LibSHTns), Ptr{Cvoid},
                   (Cint, Cint, Cint), Cint(lmax), Cint(mmax), Cint(mres))
    
    if cfg_ptr != C_NULL
        println("✅ shtns_create with explicit Cint conversion succeeded")
        
        try
            ccall((:shtns_set_grid, SHTns_jll.LibSHTns), Cvoid,
                  (Ptr{Cvoid}, Cint, Cint, Cint), cfg_ptr, Cint(nlat), Cint(nphi), Cint(0))
            println("✅ shtns_set_grid with explicit Cint conversion succeeded!")
        catch e
            println("❌ Still failed with explicit conversion: $e")
        end
        
        ccall((:shtns_free, SHTns_jll.LibSHTns), Cvoid, (Ptr{Cvoid},), cfg_ptr)
    end

catch e
    println("❌ Overall test failed: $e")
end