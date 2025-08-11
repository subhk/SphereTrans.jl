using SHTnsKit
import SHTnsKit: libshtns

println("Direct SHTns C library testing...")
println("Library path: ", libshtns)

# Step 1: Create config using direct ccall
println("\nStep 1: Creating config directly...")
cfg_ptr = ccall((:shtns_create, libshtns), Ptr{Cvoid}, (Cint, Cint, Cint), 2, 2, 1)
println("Config pointer: ", cfg_ptr)

if cfg_ptr == C_NULL
    println("ERROR: shtns_create returned NULL")
    exit(1)
end

# Step 2: Try to set grid directly
println("\nStep 2: Setting grid directly...")
println("Calling shtns_set_grid with nlat=16, nphi=17, grid_type=0")

# Use a try-catch to see exactly where the error occurs
try
    # This is the direct SHTns C call that should be causing the issue
    ccall((:shtns_set_grid, libshtns), Cvoid, (Ptr{Cvoid}, Cint, Cint, Cint), 
          cfg_ptr, 16, 17, 0)  # 0 = SHT_GAUSS
    println("shtns_set_grid succeeded!")
    
    # If we get here, the problem is elsewhere
    println("Grid setting worked - the issue must be in a later operation")
    
catch e
    println("ERROR in shtns_set_grid: ", e)
end

# Step 3: Try to free the config
println("\nStep 3: Freeing config...")
try
    ccall((:shtns_free, libshtns), Cvoid, (Ptr{Cvoid},), cfg_ptr)
    println("Config freed successfully")
catch e
    println("ERROR in shtns_free: ", e)
end