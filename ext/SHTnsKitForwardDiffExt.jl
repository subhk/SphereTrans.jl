"""
ForwardDiff Extension for Automatic Differentiation Support

This extension provides automatic differentiation capabilities for SHTnsKit using 
ForwardDiff.jl. It enables gradient computation through spherical harmonic transforms, 
which is essential for optimization problems in spherical geometry.

Key Features:
- Automatic gradient computation for scalar and vector energy functionals
- Support for both regular matrices and distributed arrays
- Seamless integration with ForwardDiff's dual number arithmetic
- Compatible with optimization workflows in geophysical modeling

Mathematical Foundation:
The extension computes gradients of energy functionals like:
- Scalar energy: E = 0.5 ∫ |f(θ,φ)|² dΩ  
- Vector energy: E = 0.5 ∫ |∇×V|² + |∇·V|² dΩ

These are fundamental quantities in fluid dynamics and field theory.
"""
module SHTnsKitForwardDiffExt

using ForwardDiff
using SHTnsKit

# ===== SCALAR FIELD GRADIENT COMPUTATION =====

"""
    fdgrad_scalar_energy(cfg, f) -> ∂E/∂f

ForwardDiff gradient of scalar energy E = 0.5 ∫ |f|^2 under spectral transform.

This function computes the functional derivative of the scalar energy with respect
to the input field f. The energy is computed in spectral space after spherical
harmonic analysis, making this useful for spectral optimization problems.

Parameters:
- cfg: SHTnsKit configuration defining the transform parameters
- f: Input scalar field matrix [nlat × nlon]

Returns:
- Gradient matrix of same size as f, representing ∂E/∂f at each point
"""
function SHTnsKit.fdgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, f::AbstractMatrix)
    nlat, nlon = size(f)
    
    # Define the energy functional as a function of flattened field
    loss(x) = SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, reshape(x, nlat, nlon)))
    
    # Use ForwardDiff to compute gradient via dual numbers
    g = ForwardDiff.gradient(loss, vec(f))
    return reshape(g, nlat, nlon)
end

##########
# Generic distributed/array wrappers (avoid hard dependency on PencilArrays)
##########

function SHTnsKit.fdgrad_scalar_energy(cfg::SHTnsKit.SHTConfig, fθφ::AbstractArray)
    nlat = length(axes(fθφ, 1)); nlon = length(axes(fθφ, 2))
    function loss_flat(z)
        xloc = reshape(z, nlat, nlon)
        return SHTnsKit.energy_scalar(cfg, SHTnsKit.analysis(cfg, xloc))
    end
    g = ForwardDiff.gradient(loss_flat, vec(Array(fθφ)))
    gl = reshape(g, nlat, nlon)
    gout = similar(fθφ)
    copyto!(gout, gl)
    return gout
end

function SHTnsKit.fdgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vtθφ::AbstractArray, Vpθφ::AbstractArray)
    nlat = length(axes(Vtθφ, 1)); nlon = length(axes(Vtθφ, 2))
    function loss_flat(z)
        Xt = reshape(view(z, 1:nlat*nlon), nlat, nlon)
        Xp = reshape(view(z, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg, Xt, Xp)
        return SHTnsKit.energy_vector(cfg, Slm, Tlm)
    end
    z0 = vcat(vec(Array(Vtθφ)), vec(Array(Vpθφ)))
    g = ForwardDiff.gradient(loss_flat, z0)
    gVt = reshape(view(g, 1:nlat*nlon), nlat, nlon)
    gVp = reshape(view(g, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
    GVt = similar(Vtθφ); GVp = similar(Vpθφ)
    copyto!(GVt, gVt); copyto!(GVp, gVp)
    return GVt, GVp
end

"""
    fdgrad_vector_energy(cfg, Vt, Vp) -> (∂E/∂Vt, ∂E/∂Vp)
"""
function SHTnsKit.fdgrad_vector_energy(cfg::SHTnsKit.SHTConfig, Vt::AbstractMatrix, Vp::AbstractMatrix)
    nlat, nlon = size(Vt)
    function loss_flat(z)
        Xt = reshape(view(z, 1:nlat*nlon), nlat, nlon)
        Xp = reshape(view(z, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
        Slm, Tlm = SHTnsKit.spat_to_SHsphtor(cfg, Xt, Xp)
        return SHTnsKit.energy_vector(cfg, Slm, Tlm)
    end
    z0 = vcat(vec(Vt), vec(Vp))
    g = ForwardDiff.gradient(loss_flat, z0)
    gVt = reshape(view(g, 1:nlat*nlon), nlat, nlon)
    gVp = reshape(view(g, nlat*nlon+1:2*nlat*nlon), nlat, nlon)
    return gVt, gVp
end

end # module
