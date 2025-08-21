#!/usr/bin/env julia

# Simple test to demonstrate the flexible library path functionality
# This can be run independently to test the feature

println("Testing SHTnsKit flexible library path functionality...")

# Test 1: Default behavior (should try SHTns_jll then fall back)
println("\n=== Test 1: Default behavior ===")
try
    # Simulate the library loading logic
    custom_lib = get(ENV, "SHTNS_LIBRARY_PATH", nothing)
    if custom_lib !== nothing
        println("Would use custom library: $custom_lib")
    else
        println("No custom library set, would try SHTns_jll then fallback to 'libshtns'")
    end
catch e
    println("Error in default test: $e")
end

# Test 2: Custom library path via environment variable
println("\n=== Test 2: Custom library via ENV ===")
try
    ENV["SHTNS_LIBRARY_PATH"] = "/custom/path/to/libshtns.so"
    custom_lib = get(ENV, "SHTNS_LIBRARY_PATH", nothing)
    if custom_lib !== nothing
        println("Custom library path detected: $custom_lib")
    end
    # Clean up
    delete!(ENV, "SHTNS_LIBRARY_PATH")
catch e
    println("Error in custom path test: $e")
end

# Test 3: Function-based approach (would be available after loading SHTnsKit)
println("\n=== Test 3: Function-based approach ===")
println("After loading SHTnsKit, users can call:")
println("  SHTnsKit.set_library_path(\"/path/to/custom/libshtns.so\")")
println("  current_path = SHTnsKit.get_library_path()")

println("\n=== Summary ===")
println("The implementation provides three ways to use custom libshtns:")
println("1. Set ENV[\"SHTNS_LIBRARY_PATH\"] before loading SHTnsKit")
println("2. Use SHTnsKit.set_library_path() after loading (requires restart)")
println("3. Falls back to SHTns_jll or system 'libshtns' if no custom path set")
println("Priority order: Custom path > SHTns_jll > System library")