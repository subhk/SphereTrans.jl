using Test
using Libdl
using SHTnsKit_jll

@testset "SHTnsKit_jll loads and basic ccall" begin
    # library path should resolve
    libpath = String(SHTnsKit_jll.libshtns_omp)
    @test isfile(libpath)

    handle = Libdl.dlopen(libpath, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    @test handle != C_NULL
    try
        has_with_opts = Libdl.dlsym_e(handle, :shtns_create_with_opts) != C_NULL
        has_create = Libdl.dlsym_e(handle, :shtns_create) != C_NULL

        if has_with_opts
            cfg = ccall((:shtns_create_with_opts, SHTnsKit_jll.libshtns_omp), Ptr{Cvoid},
                        (Cint, Cint, Cint, UInt32), 4, 4, 1, UInt32(0))
            @test cfg != C_NULL
            if Libdl.dlsym_e(handle, :shtns_destroy) != C_NULL
                ccall((:shtns_destroy, SHTnsKit_jll.libshtns_omp), Cvoid, (Ptr{Cvoid},), cfg)
            elseif Libdl.dlsym_e(handle, :shtns_free) != C_NULL
                ccall((:shtns_free, SHTnsKit_jll.libshtns_omp), Cvoid, (Ptr{Cvoid},), cfg)
            end
        elseif has_create
            # Older API variant
            cfg = ccall((:shtns_create, SHTnsKit_jll.libshtns_omp), Ptr{Cvoid}, (Cint, Cint, Cint), 4, 4, 1)
            @test cfg != C_NULL
            if Libdl.dlsym_e(handle, :shtns_destroy) != C_NULL
                ccall((:shtns_destroy, SHTnsKit_jll.libshtns_omp), Cvoid, (Ptr{Cvoid},), cfg)
            elseif Libdl.dlsym_e(handle, :shtns_free) != C_NULL
                ccall((:shtns_free, SHTnsKit_jll.libshtns_omp), Cvoid, (Ptr{Cvoid},), cfg)
            end
        else
            @info "No known create symbol found; skipping ccall test"
        end
    finally
        Libdl.dlclose(handle)
    end
end

