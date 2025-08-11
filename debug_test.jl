using SHTnsKit
import SHTnsKit: SHTnsFlags, create_config, libshtns
import Libdl

println("Testing alternative SHTns initialization...")

# Try shtns_init instead of shtns_set_grid
cfg = create_config(2, 2, 1, UInt32(0))
println("Config created: ", cfg.ptr != C_NULL)

try
    # Check what symbols are available
    handle = Libdl.dlopen(libshtns, Libdl.RTLD_LAZY)
    has_init = Libdl.dlsym_e(handle, :shtns_init) != C_NULL
    has_init_gauss = Libdl.dlsym_e(handle, :shtns_init_sh_gauss_) != C_NULL
    println("shtns_init available: ", has_init)
    println("shtns_init_sh_gauss_ available: ", has_init_gauss)
    Libdl.dlclose(handle)
    
    if has_init_gauss
        # Try the Fortran-style init function
        println("Trying shtns_init_sh_gauss_ with nlat=16, nphi=17")
        
        # This might be a Fortran function that expects specific calling convention
        # Let's try different parameter combinations
        ccall((:shtns_init_sh_gauss_, libshtns), Cvoid,
              (Ptr{Cvoid}, Cint, Cint), cfg.ptr, 16, 17)
        println("shtns_init_sh_gauss_ succeeded")
    elseif has_init
        println("Trying basic shtns_init")
        ccall((:shtns_init, libshtns), Cvoid, (Ptr{Cvoid},), cfg.ptr)
        println("shtns_init succeeded")
    end
    
catch e
    println("Error with alternative init: ", e)
end