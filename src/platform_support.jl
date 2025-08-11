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
    
    # Based on successful SHTns.jl CI evidence, all major platforms should be supported
    # However, SHTns_jll binary distribution may still have issues
    if kernel == :Darwin  # macOS
        if arch == :x86_64 || arch == :aarch64
            return :supported  # SHTns.jl shows this works
        end
    elseif kernel == :Linux
        return :supported
    elseif kernel == :Windows  
        return :supported  # SHTns.jl shows this works
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
        
        Your platform ($platform) may have compatibility issues with SHTns_jll binaries.
        However, SHTns functionality should work on all major platforms.
        
        If you encounter "SHTns accuracy test failed" errors, try:
        1. Using create_test_config() for testing/development
        2. Compiling SHTns from source and setting SHTNS_LIBRARY_PATH
        3. Reporting issues to help improve SHTns_jll compatibility
        
        Reference: SHTns.jl shows successful CI on macOS and Windows
        """
    elseif support == :unsupported
        platform = get_platform_description()
        @info """
        Platform: $platform
        
        SHTnsKit.jl should work on this platform, but testing may be limited.
        Please report any issues to help improve compatibility.
        """
    end
    
    return support
end