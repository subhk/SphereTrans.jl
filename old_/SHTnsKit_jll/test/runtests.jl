using Test
using Libdl
using SHTnsKit_jll

let ok = true
    try
        _ = SHTnsKit_jll.artifact_dir()
    catch
        ok = false
    end
    if Sys.isapple() && Sys.ARCH === :aarch64 && ok
        for lib in ("libshtns.so", "libshtns_omp.so")
            p = SHTnsKit_jll.libpath(lib)
            @test isfile(p)
            h = Libdl.dlopen_e(p)
            @test h !== nothing
            h !== nothing && Libdl.dlclose(h)
        end
    else
        @info "Skipping dlopen tests on this platform" arch=Sys.ARCH os=Sys.KERNEL artifact_bound=ok
        @test true
    end
end
