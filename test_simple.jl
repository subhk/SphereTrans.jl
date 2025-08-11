#!/usr/bin/env julia

# Simple test runner that doesn't require package installation
# This script can be used to verify the test structure works

println("SHTnsKit.jl Test Structure Verification")
println("=" ^ 40)

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Test that all test files can be parsed
test_files = [
    "test/test_basic.jl",
    "test/test_vector.jl", 
    "test/test_complex.jl",
    "test/test_rotation.jl",
    "test/test_threading.jl",
    "test/test_gpu.jl",
    "test/test_mpi.jl",
    "test/test_benchmarks.jl"
]

println("Checking test file syntax...")
for test_file in test_files
    file_path = joinpath(@__DIR__, test_file)
    if isfile(file_path)
        try
            # Just check if the file parses correctly
            ast = Base.parse_input_line(read(file_path, String))
            if ast isa Expr
                println("✓ $test_file - syntax OK")
            else
                println("✗ $test_file - parsing issue")
            end
        catch e
            println("✗ $test_file - ERROR: $e")
        end
    else
        println("✗ $test_file - file not found")
    end
end

# Check main test runner
main_test = joinpath(@__DIR__, "test/runtests.jl")
if isfile(main_test)
    try
        ast = Base.parse_input_line(read(main_test, String))
        if ast isa Expr
            println("✓ test/runtests.jl - syntax OK")
        else
            println("✗ test/runtests.jl - parsing issue")
        end
    catch e
        println("✗ test/runtests.jl - ERROR: $e")
    end
else
    println("✗ test/runtests.jl - file not found")
end

println("\nTest structure verification completed!")
println("Note: This only checks syntax, not functionality.")
println("To run actual tests, use: julia --project=. test/runtests.jl")
println("(requires SHTns C library to be installed)")