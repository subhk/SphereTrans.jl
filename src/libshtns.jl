"""
Lightweight loader utilities for the SHTns C library.

This module does not bind SHTns APIs yet; it only provides
library resolution helpers so users can point to a custom
`libshtns` or rely on the `SHTns_jll` artifact when available.

Exported API:
- `set_library_path(path::AbstractString)`
- `get_library_path()::Union{String,Nothing}`
- `validate_library()::Bool`

Resolution order:
1. Explicit path set via `set_library_path` or `ENV["SHTNS_LIBRARY_PATH"]`
2. `SHTns_jll` artifact (if available)
3. System library via `Libdl.dlopen_e("libshtns")`
"""

module LibSHTns

using Libdl

# Optional dependency; guard access at runtime
const _has_jll = Base.find_package("SHTns_jll") !== nothing

# State
const _explicit_path = Ref{Union{Nothing,String}}(get(ENV, "SHTNS_LIBRARY_PATH", nothing))
const _lib_handle = Ref{Union{Nothing,Ptr{Cvoid}}}(nothing)

"""
    set_library_path(path::AbstractString)

Set an explicit path to the `libshtns` dynamic library. Takes effect on
next validation/load. The path is not validated immediately.
"""
function set_library_path(path::AbstractString)
    _explicit_path[] = String(path)
    # Reset cached handle so a future validation reloads
    _lib_handle[] = nothing
    return _explicit_path[]
end

"""
    get_library_path() -> Union{String,Nothing}

Return the currently configured library path if explicitly set, otherwise
`nothing` (the loader will use JLL or system library).
"""
get_library_path() = _explicit_path[]

"""
    _resolve_library() -> Union{Ptr{Cvoid},Nothing}

Try to open a handle to libshtns using the configured resolution order.
Returns a handle or `nothing` if all attempts fail.
"""
function _resolve_library()
    # 1) Explicit path
    if _explicit_path[] !== nothing
        h = Libdl.dlopen_e(_explicit_path[])
        h !== C_NULL && return h
    end

    # 2) JLL artifact
    if _has_jll
        try
            @eval begin
                import SHTns_jll
            end
            # SHTns_jll exposes `libshtns` (a `JLLWrappers.LibraryProduct`)
            lib = getfield(@__MODULE__, :SHTns_jll).libshtns
            # Convert to a raw handle by opening the resolved path
            # The LibraryProduct prints to a path string when interpolated
            path = String(lib)
            h = Libdl.dlopen_e(path)
            h !== C_NULL && return h
        catch
            # ignore and fall through
        end
    end

    # 3) System library
    h = Libdl.dlopen_e("libshtns")
    h !== C_NULL && return h

    return nothing
end

"""
    validate_library() -> Bool

Attempt to load the SHTns library and check that a known symbol exists.
This does not perform any transform; it only verifies that the library
is reachable and looks like SHTns.
"""
function validate_library()
    # Reuse cached handle if available
    local h = _lib_handle[]
    if h === nothing
        h = _resolve_library()
        _lib_handle[] = h
    end
    h === nothing && return false
    # Probe for a canonical symbol present in SHTns
    # We check multiple candidates to be tolerant across versions.
    candidates = (
        :shtns_init,
        :spat_to_SH,
        :SH_to_spat,
    )
    for sym in candidates
        ptr = Libdl.dlsym_e(h, sym)
        if ptr !== C_NULL
            return true
        end
    end
    return false
end

end # module LibSHTns

