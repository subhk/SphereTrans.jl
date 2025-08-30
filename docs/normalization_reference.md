# SHTnsKit Normalization Reference

This document explains the normalization conventions implemented in SHTnsKit.jl, corresponding to the SHTns C library standards.

## Normalization Conventions

### 1. SHT_ORTHONORMAL (Default)
**Description**: Orthonormal spherical harmonics with 4π normalization
**Formula**: Y_l^m has unit norm over the unit sphere
**Integration Factor**: 4π (for analysis from spatial data)
**Real Coefficients**: Factor of 2 for m > 0 (real/imaginary separation)

### 2. SHT_FOURPI 
**Description**: 4π normalization convention
**Formula**: Coefficients scaled by (2l+1)/4π
**Integration Factor**: 4π
**Synthesis**: Includes factorial corrections for proper reconstruction

### 3. SHT_SCHMIDT
**Description**: Schmidt semi-normalized harmonics
**Formula**: Removes sqrt(2) factor for m > 0 terms
**Integration Factor**: 4π with m-dependent corrections
**Usage**: Common in geophysics applications

### 4. SHT_REAL_NORM
**Description**: Real-valued normalization for unit sphere
**Formula**: Standard normalization without complex factors
**Integration Factor**: 4π

## Implementation Details

### Analysis Transform Normalization
```julia
function _get_analysis_normalization(cfg, l, m)
```
- Converts spatial data to spectral coefficients
- Accounts for quadrature weights and grid integration
- Handles real/complex coefficient separation for m > 0

### Synthesis Transform Normalization  
```julia
function _get_synthesis_normalization(cfg, l, m)
```
- Reconstructs spatial data from spectral coefficients
- Applies degree-dependent scaling factors
- Ensures proper amplitude preservation

### Vector Field Normalization
```julia
function _get_vector_analysis_normalization(cfg, l, m)
```
- Includes l(l+1) scaling for vector spherical harmonics
- Accounts for gradient operations in vector transforms
- Handles both toroidal and spheroidal components

## Mathematical Background

The spherical harmonic functions Y_l^m(θ,φ) satisfy:

∫∫ Y_l^m(θ,φ) Y_{l'}^{m'}*(θ,φ) sin(θ) dθ dφ = δ_{ll'} δ_{mm'} N_{lm}

Where N_{lm} is the normalization constant depending on the convention:

- **Orthonormal**: N_{lm} = 1
- **4π**: N_{lm} = 4π/(2l+1) 
- **Schmidt**: N_{lm} = 4π with m-dependent factors
- **Real**: N_{lm} = 4π/(2l+1) without complex factors

## Usage Examples

```julia
# Create configuration with orthonormal harmonics
cfg = create_config(Float64, 10, 10, 1; norm=SHT_ORTHONORMAL)  # nlat auto-adjusted if < lmax+1

# Analysis normalization for l=5, m=3
analysis_norm = _get_analysis_normalization(cfg, 5, 3)

# Synthesis normalization for reconstruction
synthesis_norm = _get_synthesis_normalization(cfg, 5, 3)

# Vector field normalization
vector_norm = _get_vector_analysis_normalization(cfg, 5, 3)
```

## Notes

1. **Consistency**: All normalization functions are consistent with SHTns C library
2. **Performance**: Optimized implementations cache common factors
3. **Accuracy**: Proper handling of factorial terms and m-dependent corrections
4. **Flexibility**: Support for all major normalization conventions
