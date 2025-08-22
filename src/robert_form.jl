"""
Robert form support for vector transforms.
In Robert form, the vector synthesis returns fields multiplied by sin(θ),
while the analysis divides by sin(θ) before the transform.
"""

"""
    set_robert_form!(cfg::SHTnsConfig{T}, enable::Bool) where T

Enable or disable Robert form for vector transforms.

# Arguments
- `cfg`: SHTns configuration  
- `enable`: Whether to enable (true) or disable (false) Robert form

In Robert form:
- Vector synthesis: output is multiplied by sin(θ)
- Vector analysis: input is divided by sin(θ) before transform

This is useful for atmospheric and oceanic applications where the governing
equations are naturally written in terms of sin(θ)-weighted quantities.

Equivalent to the C library function `shtns_robert_form()`.
"""
function set_robert_form!(cfg::SHTnsConfig{T}, enable::Bool) where T
    validate_config(cfg)
    cfg.robert_form = enable
    return nothing
end

"""
    is_robert_form(cfg::SHTnsConfig{T}) where T

Check whether Robert form is enabled for the given configuration.

# Returns
- `true` if Robert form is enabled, `false` otherwise
"""
function is_robert_form(cfg::SHTnsConfig{T}) where T
    return cfg.robert_form
end

"""
    sphtor_to_spat_robert!(cfg::SHTnsConfig{T}, 
                          sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                          u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T

Vector synthesis with Robert form applied.
The output vector components are multiplied by sin(θ).

This is equivalent to `sphtor_to_spat!` but with the Robert form modification.
"""
function sphtor_to_spat_robert!(cfg::SHTnsConfig{T},
                               sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T},
                               u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    validate_config(cfg)
    
    # Perform regular synthesis
    sphtor_to_spat!(cfg, sph_coeffs, tor_coeffs, u_theta, u_phi)
    
    # Apply Robert form: multiply by sin(θ)
    if cfg.robert_form
        _apply_robert_form_synthesis!(cfg, u_theta, u_phi)
    end
    
    return (u_theta, u_phi)
end

"""
    spat_to_sphtor_robert!(cfg::SHTnsConfig{T},
                          u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                          sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T

Vector analysis with Robert form applied.
The input vector components are divided by sin(θ) before analysis.
"""
function spat_to_sphtor_robert!(cfg::SHTnsConfig{T},
                               u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T},
                               sph_coeffs::AbstractVector{T}, tor_coeffs::AbstractVector{T}) where T
    validate_config(cfg)
    
    if cfg.robert_form
        # Create temporary arrays for Robert form preprocessing
        u_theta_work = copy(u_theta)
        u_phi_work = copy(u_phi)
        
        # Apply Robert form: divide by sin(θ)
        _apply_robert_form_analysis!(cfg, u_theta_work, u_phi_work)
        
        # Perform analysis on modified data
        spat_to_sphtor!(cfg, u_theta_work, u_phi_work, sph_coeffs, tor_coeffs)
    else
        # Regular analysis without Robert form
        spat_to_sphtor!(cfg, u_theta, u_phi, sph_coeffs, tor_coeffs)
    end
    
    return (sph_coeffs, tor_coeffs)
end

# Internal implementation functions

"""
    _apply_robert_form_synthesis!(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T

Apply Robert form modification to synthesis output: multiply by sin(θ).
"""
function _apply_robert_form_synthesis!(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    for j in 1:nlat
        sint = sin(cfg.theta_grid[j])
        
        # Apply sin(θ) factor to both components
        for i in 1:nphi
            u_theta[j, i] *= sint
            u_phi[j, i] *= sint
        end
    end
end

"""
    _apply_robert_form_analysis!(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T

Apply Robert form modification to analysis input: divide by sin(θ).
"""
function _apply_robert_form_analysis!(cfg::SHTnsConfig{T}, u_theta::AbstractMatrix{T}, u_phi::AbstractMatrix{T}) where T
    nlat, nphi = cfg.nlat, cfg.nphi
    
    for j in 1:nlat
        sint = sin(cfg.theta_grid[j])
        
        # Avoid division by zero near poles
        if sint > 1e-12
            inv_sint = 1 / sint
            
            # Apply 1/sin(θ) factor to both components
            for i in 1:nphi  
                u_theta[j, i] *= inv_sint
                u_phi[j, i] *= inv_sint
            end
        else
            # Near poles, set values to zero to avoid infinities
            for i in 1:nphi
                u_theta[j, i] = zero(T)
                u_phi[j, i] = zero(T)
            end
        end
    end
end

"""
    robert_form_factor(cfg::SHTnsConfig{T}, j::Int) where T

Get the Robert form factor sin(θ) for the j-th latitude point.

# Arguments
- `cfg`: SHTns configuration
- `j`: Latitude index (1-based)

# Returns
- sin(θ) value for the j-th grid point
"""
function robert_form_factor(cfg::SHTnsConfig{T}, j::Int) where T
    validate_config(cfg)
    1 <= j <= cfg.nlat || error("j must be in [1, nlat]")
    
    return sin(cfg.theta_grid[j])
end

"""
    apply_robert_form_to_field!(cfg::SHTnsConfig{T}, field::AbstractMatrix{T}, synthesis::Bool=true) where T

Apply Robert form scaling to a field.

# Arguments  
- `cfg`: SHTns configuration
- `field`: Input/output field (nlat × nphi)
- `synthesis`: If true, multiply by sin(θ); if false, divide by sin(θ)

This is a utility function for applying Robert form scaling to arbitrary fields.
"""
function apply_robert_form_to_field!(cfg::SHTnsConfig{T}, field::AbstractMatrix{T}, synthesis::Bool=true) where T
    validate_config(cfg)
    size(field) == (cfg.nlat, cfg.nphi) || error("field size must be (nlat, nphi)")
    
    nlat, nphi = cfg.nlat, cfg.nphi
    
    for j in 1:nlat
        sint = sin(cfg.theta_grid[j])
        
        if synthesis
            # Synthesis: multiply by sin(θ)
            for i in 1:nphi
                field[j, i] *= sint
            end
        else
            # Analysis: divide by sin(θ)  
            if sint > 1e-12
                inv_sint = 1 / sint
                for i in 1:nphi
                    field[j, i] *= inv_sint
                end
            else
                # Near poles, set to zero
                for i in 1:nphi
                    field[j, i] = zero(T)
                end
            end
        end
    end
    
    return field
end