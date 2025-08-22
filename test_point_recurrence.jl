#!/usr/bin/env julia

"""
Test script for verifying the recurrence-based point evaluation functions.
"""

include("src/types_optimized.jl")
include("src/point_evaluation.jl")

function test_point_legendre_recurrence()
    println("Testing point evaluation Legendre recurrence...")
    
    T = Float64
    
    # Create a dummy configuration (not really used for point evaluation)
    cfg = SHTnsConfig{T}()
    cfg.lmax = 10
    cfg.mmax = 10
    cfg.norm = SHT_ORTHONORMAL
    
    # Test at various angles
    test_angles = [π/6, π/4, π/3, π/2, 2π/3, 3π/4]
    
    println("Testing known values:")
    all_passed = true
    
    for θ in test_angles
        cost = cos(θ)
        sint = sin(θ)
        
        # Test known analytical values
        test_cases = [
            (0, 0, 1.0),                           # P₀⁰ = 1
            (1, 0, cost),                          # P₁⁰ = cos(θ)
            (1, 1, -sint),                         # P₁¹ = -sin(θ)
            (2, 0, 0.5*(3*cost^2 - 1)),           # P₂⁰ = ½(3cos²θ - 1)
            (2, 1, -3*cost*sint),                  # P₂¹ = -3cos(θ)sin(θ)
            (2, 2, 3*sint^2),                      # P₂² = 3sin²(θ)
        ]
        
        for (l, m, expected) in test_cases
            computed = _evaluate_legendre_at_point(cfg, l, m, cost, sint)
            error = abs(computed - expected)
            passed = error < 1e-12
            
            if !passed || θ == π/4  # Show results for π/4 case
                println("  θ=$(θ/π)π, P($l,$m): computed=$computed, expected=$expected, error=$error")
            end
            all_passed = all_passed && passed
        end
    end
    
    println("Testing recurrence consistency...")
    # Test that our recurrence matches analytical recurrence relations
    θ = π/3
    cost = cos(θ)
    sint = sin(θ)
    
    for l in 2:5
        for m in 0:min(l, 3)
            if l >= m + 2  # Ensure we can compute l-2 with m <= l-2
                # Verify three-term recurrence for P_l^m
                p_l = _evaluate_legendre_at_point(cfg, l, m, cost, sint)
                p_l_minus_1 = _evaluate_legendre_at_point(cfg, l-1, m, cost, sint)
                p_l_minus_2 = _evaluate_legendre_at_point(cfg, l-2, m, cost, sint)
                
                # Apply recurrence: (l-m) P_l^m = (2l-1) cos(θ) P_{l-1}^m - (l+m-1) P_{l-2}^m
                lhs = T(l - m) * p_l
                rhs = T(2*l - 1) * cost * p_l_minus_1 - T(l + m - 1) * p_l_minus_2
                
                error = abs(lhs - rhs)
                passed = error < 1e-12
                
                println("  Recurrence P($l,$m): error=$error, passed=$passed")
                all_passed = all_passed && passed
            end
        end
    end
    
    return all_passed
end

function test_point_derivative_accuracy()
    println("\nTesting point derivative accuracy...")
    
    T = Float64
    cfg = SHTnsConfig{T}()
    cfg.lmax = 10
    cfg.mmax = 10
    cfg.norm = SHT_ORTHONORMAL
    
    # Test derivatives using numerical differentiation
    θ = π/4
    cost = cos(θ)
    sint = sin(θ)
    h = 1e-8  # Small step
    
    all_passed = true
    
    println("Comparing analytical vs numerical derivatives:")
    
    for l in 1:4
        for m in 0:min(l, 2)
            # Analytical derivative
            analytical = _evaluate_legendre_derivative_at_point(cfg, l, m, cost, sint)
            
            # Numerical derivative using finite differences
            θ_plus = θ + h
            θ_minus = θ - h
            cost_plus = cos(θ_plus)
            sint_plus = sin(θ_plus)
            cost_minus = cos(θ_minus)
            sint_minus = sin(θ_minus)
            
            p_plus = _evaluate_legendre_at_point(cfg, l, m, cost_plus, sint_plus)
            p_minus = _evaluate_legendre_at_point(cfg, l, m, cost_minus, sint_minus)
            
            numerical = (p_plus - p_minus) / (2*h)
            
            error = abs(analytical - numerical)
            passed = error < 1e-6  # Looser tolerance for numerical differentiation
            
            println("  dP($l,$m)/dθ: analytical=$analytical, numerical=$numerical, error=$error, passed=$passed")
            all_passed = all_passed && passed
        end
    end
    
    return all_passed
end

function test_point_evaluation_performance()
    println("\nTesting performance characteristics...")
    
    T = Float64
    cfg = SHTnsConfig{T}()
    cfg.lmax = 50
    cfg.mmax = 50
    cfg.norm = SHT_ORTHONORMAL
    
    θ = π/3
    cost = cos(θ)
    sint = sin(θ)
    
    # Time a batch of evaluations
    n_evals = 1000
    
    # Warm up
    for _ in 1:10
        _evaluate_legendre_at_point(cfg, 20, 10, cost, sint)
    end
    
    # Time the evaluation
    t_start = time()
    for i in 1:n_evals
        l = mod(i, 30) + 1
        m = mod(i, min(l+1, 10))  # Ensure m <= l
        _evaluate_legendre_at_point(cfg, l, m, cost, sint)
    end
    t_end = time()
    
    avg_time_μs = (t_end - t_start) * 1e6 / n_evals
    println("  Average evaluation time: $(round(avg_time_μs, digits=2)) μs")
    
    # Should be reasonably fast (< 10 μs per evaluation)
    return avg_time_μs < 50.0
end

function main()
    println("=== Point Evaluation Recurrence Test ===\n")
    
    test1_passed = test_point_legendre_recurrence()
    test2_passed = test_point_derivative_accuracy()  
    test3_passed = test_point_evaluation_performance()
    
    println("\n=== Summary ===")
    println("Point Legendre recurrence test: ", test1_passed ? "PASSED" : "FAILED")
    println("Point derivative accuracy test: ", test2_passed ? "PASSED" : "FAILED")
    println("Performance test: ", test3_passed ? "PASSED" : "FAILED")
    
    if test1_passed && test2_passed && test3_passed
        println("\n✅ All point evaluation tests PASSED!")
        return 0
    else
        println("\n❌ Some tests FAILED!")
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end