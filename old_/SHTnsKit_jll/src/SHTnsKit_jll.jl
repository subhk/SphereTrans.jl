module SHTnsKit_jll

using Artifacts
using Libdl

const _artifact_name = "SHTnsKit"

"""
Return the absolute path to the root of the SHTnsKit artifact directory.
This will trigger a download on first use if not already present.
"""
function artifact_dir()
    return Artifacts.@artifact_str _artifact_name
end

"""
Return the absolute path to a library inside the artifact's lib/ directory.
"""
libpath(libname::AbstractString) = joinpath(artifact_dir(), "lib", libname)

"""
dlopen both libs to validate availability at init time.
Closes immediately; callers can `Libdl.dlopen` again if needed.
"""
function __init__()
    # Only attempt dlopen on supported platforms for which we provide an artifact
    if !Sys.isapple()
        return
    end

    # Attempt to locate the artifact (may download on first use)
    local _root
    try
        _root = artifact_dir()
    catch e
        @info "Artifact not available; skipping dlopen" arch=Sys.ARCH os=Sys.KERNEL exception=e
        return
    end
    if true
        # Try common library filenames for Apple platforms
        local candidates
        candidates = [
            "libshtns.dylib", "libshtns_omp.dylib",
            "libshtns.so",    "libshtns_omp.so",
        ]
        for lib in candidates
            p = joinpath(_root, "lib", lib)
            if !isfile(p)
                # Skip missing names; different builds may choose .dylib vs .so
                continue
            end
            h = Libdl.dlopen_e(p)
            if h === nothing
                @error "Failed to dlopen library" lib=p
            else
                Libdl.dlclose(h)
            end
        end
    else
        @info "Skipping dlopen on unsupported platform" arch=Sys.ARCH os=Sys.KERNEL
    end
end

end # module
