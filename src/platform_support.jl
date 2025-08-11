"""
Platform-specific support detection for SHTnsKit.jl

This module detects known issues with SHTns_jll on different platforms
and provides appropriate warnings or workarounds.
"""

"""
Check if the current platform has known issues with SHTns_jll.

Returns:
- `:supported` - Platform works reliably
- `:problematic` - Known issues, but may work with workarounds  
- `:unsupported` - Major compatibility problems
"""
function check_platform_support()
    kernel = Sys.KERNEL
    arch = Sys.ARCH
    
    # Check for known problematic combinations
    if kernel == :Darwin  # macOS
        if arch == :x86_64 || arch == :aarch64
            return :problematic
        end
    elseif kernel == :Linux
        return :supported
    elseif kernel == :Windows  
        return :supported  # Assume supported unless proven otherwise
    end
    
    return :problematic  # Conservative default for unknown platforms
end

"""
Get a user-friendly platform description.
"""
function get_platform_description()
    kernel = Sys.KERNEL
    arch = Sys.ARCH
    
    if kernel == :Darwin
        if arch == :aarch64
            return "macOS (Apple Silicon)"
        else
            return "macOS (Intel)"
        end
    elseif kernel == :Linux
        return "Linux ($arch)"
    elseif kernel == :Windows
        return "Windows ($arch)"
    else
        return "$kernel ($arch)"
    end
end

"""
Issue a warning if the platform has known SHTns_jll issues.
"""
function warn_if_problematic_platform()
    support = check_platform_support()
    
    if support == :problematic
        platform = get_platform_description()
        @warn """
        Platform compatibility notice for SHTnsKit.jl:
        
        Your platform ($platform) has known compatibility issues with SHTns_jll.
        You may encounter "SHTns accuracy test failed" errors.
        
        For reliable operation, consider:
        1. Using GitHub Actions with 'ubuntu-latest' for CI/CD
        2. Running in a Linux Docker container
        3. Compiling SHTns from source and setting SHTNS_LIBRARY_PATH
        
        If you encounter errors, they are likely platform-related, not code issues.
        """
    elseif support == :unsupported
        platform = get_platform_description()
        @warn """
        Unsupported platform detected: $platform
        
        SHTnsKit.jl with SHTns_jll has not been tested on this platform.
        Consider using a Linux environment for reliable operation.
        """
    end
    
    return support
end