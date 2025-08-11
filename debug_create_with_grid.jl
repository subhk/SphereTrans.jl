using SHTnsKit
import SHTnsKit: libshtns, SHTnsFlags

println("Testing shtns_create_with_grid...")
println("Library path: ", libshtns)

# Try the combined create_with_grid function instead
println("\nTrying shtns_create_with_grid...")
println("Parameters: lmax=2, mmax=2, mres=1, nlat=16, nphi=17, grid_type=0")

try
    cfg_ptr = ccall((:shtns_create_with_grid, libshtns), Ptr{Cvoid}, 
                   (Cint, Cint, Cint, Cint, Cint, Cint), 
                   2, 2, 1, 16, 17, 0)  # lmax, mmax, mres, nlat, nphi, grid_type
    
    if cfg_ptr == C_NULL
        println("ERROR: shtns_create_with_grid returned NULL")
    else
        println("SUCCESS: shtns_create_with_grid returned: ", cfg_ptr)
        
        # Try to free it
        ccall((:shtns_free, libshtns), Cvoid, (Ptr{Cvoid},), cfg_ptr)
        println("Config freed successfully")
    end
    
catch e
    println("ERROR in shtns_create_with_grid: ", e)
end

# Also try with different parameters
println("\nTrying with larger grid...")
try
    cfg_ptr = ccall((:shtns_create_with_grid, libshtns), Ptr{Cvoid}, 
                   (Cint, Cint, Cint, Cint, Cint, Cint), 
                   4, 4, 1, 32, 33, 0)  # lmax, mmax, mres, nlat, nphi, grid_type
    
    if cfg_ptr == C_NULL
        println("ERROR: shtns_create_with_grid returned NULL with larger params")
    else
        println("SUCCESS: shtns_create_with_grid with larger params: ", cfg_ptr)
        ccall((:shtns_free, libshtns), Cvoid, (Ptr{Cvoid},), cfg_ptr)
    end
    
catch e
    println("ERROR with larger params: ", e)
end