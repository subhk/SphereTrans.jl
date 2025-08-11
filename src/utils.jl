"""
    set_library_path(path::AbstractString)

Set a custom path to the libshtns shared library. This allows users to use their own 
compiled version of SHTns instead of relying on SHTns_jll.

# Arguments
- `path`: Full path to the libshtns shared library file

# Examples
```julia
# Set custom library path
SHTnsKit.set_library_path("/usr/local/lib/libshtns.so")        # Linux
SHTnsKit.set_library_path("/usr/local/lib/libshtns.dylib")     # macOS
SHTnsKit.set_library_path("C:/lib/libshtns.dll")               # Windows

# Or use environment variable before loading SHTnsKit
ENV["SHTNS_LIBRARY_PATH"] = "/usr/local/lib/libshtns.so"
using SHTnsKit
```

# Notes
- This must be called before any SHTns functions are used
- The library path can also be set via the SHTNS_LIBRARY_PATH environment variable
- Environment variable takes precedence over SHTns_jll
- The function validates that the library file exists and contains SHTns symbols
"""
function set_library_path(path::AbstractString)
    # Expand path to handle ~ and relative paths
    expanded_path = expanduser(path)
    absolute_path = abspath(expanded_path)
    
    if !isfile(absolute_path)
        error("Library file not found: $absolute_path")
    end
    
    # Validate it's actually a SHTns library
    try
        handle = Libdl.dlopen(absolute_path, Libdl.RTLD_LAZY)
        has_shtns = Libdl.dlsym_e(handle, :shtns_create_with_opts) != C_NULL
        Libdl.dlclose(handle)
        
        if !has_shtns
            error("Library $absolute_path does not appear to be a valid SHTns library (missing shtns_create_with_opts symbol)")
        end
    catch e
        if isa(e, SystemError) && e.errnum == Base.UV_ENOENT
            error("Library file could not be loaded: $absolute_path. Check if the file is a valid shared library.")
        else
            rethrow(e)
        end
    end
    
    ENV["SHTNS_LIBRARY_PATH"] = absolute_path
    @warn "Library path set to: $absolute_path. You may need to restart Julia for this to take effect."
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

"""
    validate_library()

Validate that the current libshtns library is accessible and contains required SHTns symbols.

# Returns
- `true` if the library is valid and accessible
- `false` if the library has issues

# Examples
```julia
if SHTnsKit.validate_library()
    println("SHTns library is working correctly")
else
    println("There may be issues with the SHTns library")
end
```
"""
function validate_library()
    try
        # Try to open the library
        handle = Libdl.dlopen(libshtns, Libdl.RTLD_LAZY)
        
        # Check for key SHTns symbols
        required_symbols = [:shtns_create_with_opts, :shtns_set_grid, :shtns_sh_to_spat, :shtns_spat_to_sh]
        for sym in required_symbols
            if Libdl.dlsym_e(handle, sym) == C_NULL
                @error "Missing required SHTns symbol: $sym"
                Libdl.dlclose(handle)
                return false
            end
        end
        
        Libdl.dlclose(handle)
        return true
    catch e
        @error "Failed to validate SHTns library: $e"
        return false
    end
end

"""
    find_system_library() -> Union{String, Nothing}

Attempt to find SHTns library in common system locations.

# Returns
- Path to found library or `nothing` if not found

# Examples
```julia
lib_path = SHTnsKit.find_system_library()
if lib_path !== nothing
    println("Found SHTns library at: $lib_path")
    SHTnsKit.set_library_path(lib_path)
end
```
"""
function find_system_library()
    # Common library names across platforms
    if Sys.iswindows()
        lib_names = ["libshtns.dll", "shtns.dll"]
        search_paths = [
            "C:/lib", "C:/usr/lib", "C:/Program Files/lib",
            joinpath(get(ENV, "USERPROFILE", ""), "lib")
        ]
    elseif Sys.isapple()
        lib_names = ["libshtns.dylib", "libshtns.so"]
        search_paths = [
            "/usr/local/lib", "/opt/homebrew/lib", "/usr/lib",
            "/opt/local/lib", joinpath(homedir(), ".local/lib")
        ]
    else  # Linux and other Unix-like
        lib_names = ["libshtns.so", "libshtns.so.1", "libshtns.so.0"]
        search_paths = [
            "/usr/local/lib", "/usr/lib", "/lib",
            "/usr/lib/x86_64-linux-gnu", "/usr/local/lib64", "/usr/lib64",
            joinpath(homedir(), ".local/lib")
        ]
    end
    
    # Search in common locations
    for search_path in search_paths, lib_name in lib_names
        candidate = joinpath(search_path, lib_name)
        if isfile(candidate)
            # Quick validation
            try
                handle = Libdl.dlopen(candidate, Libdl.RTLD_LAZY)
                has_shtns = Libdl.dlsym_e(handle, :shtns_create_with_opts) != C_NULL
                Libdl.dlclose(handle)
                if has_shtns
                    return candidate
                end
            catch
                continue  # Try next candidate
            end
        end
    end
    
    return nothing
end
