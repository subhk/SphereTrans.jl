# SIMD Vectorization Safety Guide for SHTnsKit.jl

## Overview

This guide documents safe vs unsafe SIMD vectorization patterns in SHTnsKit.jl, focusing on the critical distinction between `@simd` and `@simd ivdep` annotations.

## Key Rules

### ✅ Safe for `@simd ivdep`
- **Independent array writes**: Each iteration writes to a different array element
- **No iteration dependencies**: Current iteration doesn't depend on previous ones
- **No reductions**: Not accumulating into the same variable

### ❌ Unsafe for `@simd ivdep` (Use `@simd` instead)  
- **Reduction operations**: Accumulating into same array elements across iterations
- **Iteration dependencies**: Current value depends on previous iterations
- **Recurrence relations**: Each step requires results from previous steps

## Safe Patterns

### 1. Independent Element Updates
```julia
# ✅ SAFE - each i writes to different Fφ[i,col]
@simd ivdep for i in 1:nlat
    Fφ[i, col] = inv_scaleφ * G[i]
end
```

### 2. Normalization Loops
```julia  
# ✅ SAFE - each (l,m) pair is independent
@simd ivdep for l in m:lmax
    alm[l+1, col] *= cfg.Nlm[l+1, col] * scaleφ
end
```

### 3. Array Format Conversions
```julia
# ✅ SAFE - independent element-wise operations
@simd ivdep for l in m:lmax
    packed[idx] = dense[l+1, m+1]
    idx += 1
end
```

### 4. Derivative Computations
```julia
# ✅ SAFE - each dPdx[l+1] computed from independent P[l] and P[l+1]
@simd ivdep for l in (m+1):lmax
    dPdx[l+1] = (l * x * P[l+1] - (l + m) * P[l]) / x2m1
end
```

## Unsafe Patterns (Fixed in Codebase)

### 1. Accumulation/Reduction Operations
```julia
# ❌ UNSAFE - accumulating into same alm[l+1,col] across i iterations
for i in 1:nlat
    @simd ivdep for l in m:lmax  # WRONG!
        alm[l+1, col] += wi * P[l+1] * Fi
    end
end

# ✅ FIXED - use @simd for reductions
for i in 1:nlat
    @simd for l in m:lmax  # Compiler can optimize reductions
        alm[l+1, col] += wi * P[l+1] * Fi  
    end
end
```

### 2. Recurrence Relations  
```julia
# ❌ UNSAFE - P[l+1] depends on P[l] and P[l-1]
@simd ivdep for l in 2:lmax  # WRONG!
    P[l+1] = ((2l-1) * x * P[l] - (l-1) * P[l-1]) / l
end

# ✅ FIXED - removed @simd entirely (cannot vectorize)
for l in 2:lmax
    P[l+1] = ((2l-1) * x * P[l] - (l-1) * P[l-1]) / l
end
```

### 3. DFT/FFT Accumulation Loops
```julia
# ❌ UNSAFE - accumulating into same 's' variable
@simd ivdep for j in 0:(nlon-1)  # WRONG!
    s += A[i, j+1] * cis(dir * _TWO_PI * k * j / nlon)
end

# ✅ FIXED - use @simd for reduction optimization  
@simd for j in 0:(nlon-1)
    s += A[i, j+1] * cis(dir * _TWO_PI * k * j / nlon)
end
```

## Implementation Status

All loops in SHTnsKit.jl have been audited and corrected:

### Files Checked and Fixed:
- ✅ `src/fftutils.jl` - DFT fallback loops
- ✅ `src/legendre.jl` - Recurrence relations and derivatives  
- ✅ `src/transform.jl` - Analysis/synthesis loops
- ✅ `ext/parallel_transforms.jl` - Parallel analysis/synthesis

### Remaining Safe Uses:
- Element normalization (independent updates)
- Format conversions (packed ↔ dense storage)
- Synthesis output (independent array writes)
- Derivative calculations (no dependencies)

## Performance Impact

- **@simd ivdep**: Maximum vectorization, assumes no dependencies
- **@simd**: Allows compiler to optimize reductions and detect safe vectorization
- **No annotation**: Standard scalar execution

The fixes maintain correctness while allowing the compiler to vectorize where safe, preserving most performance benefits.