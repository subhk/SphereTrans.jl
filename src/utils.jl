"""
    set_library_path(path::AbstractString)

Set a custom path to the libshtns shared library. This allows users to use their own 
compiled version of SHTns instead of relying on SHTns_jll.

# Arguments
- `path`: Full path to the libshtns shared library file

# Examples
```julia
# Set custom library path
SHTnsKit.set_library_path("/usr/local/lib/libshtns.so")

# Or use environment variable before loading SHTnsKit
ENV["SHTNS_LIBRARY_PATH"] = "/usr/local/lib/libshtns.so"
using SHTnsKit
```

# Notes
- This must be called before any SHTns functions are used
- The library path can also be set via the SHTNS_LIBRARY_PATH environment variable
- Environment variable takes precedence over SHTns_jll
"""
function set_library_path(path::AbstractString)
    if !isfile(path)
        error("Library file not found: $path")
    end
    ENV["SHTNS_LIBRARY_PATH"] = path
    @warn "Library path set to: $path. You may need to restart Julia for this to take effect."
end

"""
    get_library_path()

Get the currently configured libshtns library path.

# Returns
- The path to the libshtns library being used
"""
function get_library_path()
    return string(libshtns)
end
