
abstract type AbstractAsymptoticPosterior{P,T,Pm1,R} <: AbstractDifferentiableObject{P,T} end

struct AsymptoticPosteriorFD{P,T,Pm1,R,D<:AbstractDifferentiableObject{Pm1,T},M,S,MAP <: MAPFD{P,T,R},
                            R2,SM <: MutableFixedSizeMatrix{R2,Pm1,T}} <: AbstractAsymptoticPosterior{P,T,Pm1,R}
    od::D
    method::M
    state::S
    map::MAP
    nlmax::Base.RefValue{T}
    rstar::Base.RefValue{T}
    α::Base.RefValue{T}
    Lsmall::SM
end

## abstract interface
## required methods

@inline set_profile_val_ind!(ap::AsymptoticPosteriorFD, v, i) = set_profile_val_ind!(ap.od.config.f, v, i)
@inline setswap!(ap::AsymptoticPosteriorFD, i) = ap.map.od.config.f.i[] = i
@inline rstar(ap::AsymptoticPosteriorFD) = ap.rstar[]
@inline rstar!(ap::AsymptoticPosteriorFD{P,T}, v::T) where {P,T} = (ap.rstar[] = v)
@inline α(ap::AsymptoticPosteriorFD) = ap.α[]
@inline function α!(ap::AsymptoticPosteriorFD{P,T}, v::T) where {P,T}
    ap.α[] = v
    rstar!(ap, Φ⁻¹(v))
    v
end
@inline mode(ap::AsymptoticPosteriorFD) = ap.map.θhat
@inline mode(ap::AsymptoticPosteriorFD, i::Integer) = @inbounds ap.map.θhat[i]
@inline std_estimates(ap::AsymptoticPosteriorFD) = ap.map.std_estimates
@inline std_estimates(ap::AsymptoticPosteriorFD, i) = @inbounds ap.map.std_estimates[i]

@inline gradient(ap::AsymptoticPosteriorFD) = ap.map.od.config.result.grad
@inline hessian(ap::AsymptoticPosteriorFD) = DiffResults.hessian(ap.map.od.config.result)

@inline profile_ind(ap::AsymptoticPosteriorFD) = ap.od.config.f.i[]
@inline profile_ind!(ap::AsymptoticPosteriorFD, i::Integer) = (ap.od.config.f.i[] = i)
@inline profile_val(ap::AsymptoticPosteriorFD) = ap.od.config.f.v[]
@inline nl_profile_max(ap::AsymptoticPosteriorFD) = ap.nlmax[]
@inline nl_profile_max!(ap::AsymptoticPosteriorFD{P,T}, v::T) where {P,T} = (ap.nlmax[] = v)
@inline nl_max(ap::AsymptoticPosteriorFD) = ap.map.nlmax[]
@inline nl_max!(ap::AsymptoticPosteriorFD{P,T}, v::T) where {P,T} = (ap.map.nlmax[] = v)

@inline fit!(ap::AsymptoticPosteriorFD{P,T}, x::AbstractFixedSizeVector{P,T}) where {P,T} = fit!(ap.map, x)

"""
initial_val_buffer(ap::AbstractAsymptoticPosterior{P,T}) where {P,T}
returns a mutable buffer of length {P}.
The buffer shuold be <: AbstractArray{T}.
Each call must refer to the same piece of memory.
"""
@inline initial_val_buffer(ap::AsymptoticPosteriorFD) = ap.map.buffer

@inline nuisance_parameters(ap::AsymptoticPosteriorFD) = DifferentiableObjects.ref_x_old(ap.state)

@inline base_adjustment(ap::AsymptoticPosteriorFD) = ap.map.base_adjust[]
@inline base_adjustment!(ap::AsymptoticPosteriorFD, v) = (ap.map.base_adjust[] = v)
@inline Lfull(ap::AsymptoticPosteriorFD) = ap.map.Lfull

function refit!(ap::AbstractAsymptoticPosterior{P}) where {P}
    i = profile_ind(ap)
    setswap!(ap, P)
    fit!(ap.map, ap.od.config.f.x)
    setswap!(ap, i)
end

@inline function delta_log_likelihood(ap::AbstractAsymptoticPosterior)#{P}) where {P}
#    return nl_profile_max(ap) - nl_max(ap)
    δ = nl_profile_max(ap) - nl_max(ap)
    if δ < 0
        refit!(ap)
        return delta_log_likelihood(ap)
#        @warn "δ log likelihood = $δ < 0, returning 0."
#        return zero(δ)
    end
    δ
end

@inline profile_lower_triangle(ap::AsymptoticPosteriorFD) = ap.Lsmall 

@inline function optimize_profile!(ap::AsymptoticPosteriorFD{P,T}) where {P,T}
    scaled_nlmax, scale = optimize_scale!(ap.state, ap.od, nuisance_parameters(ap), ap.method, one(T))
    nl_profile_max!(ap, scaled_nlmax / scale)
end

@inline function DifferentiableObjects.hessian!(ap::AsymptoticPosteriorFD)
    DifferentiableObjects.hessian!(ap.map.od.config, ap.map.buffer)
end

## 
## Abstract interface, optional methods
## These are optional, because they all have the default fallbacks, shown below.
## 

"""
Naively assumes indepedence, and just uses the global maximum for the profile maximum.
This seems like it ought to perform poorly, but in tests it appears more robust.
"""
function naive_expected_nuisance!(ap::AbstractAsymptoticPosterior{P}, i::Int = profile_ind(ap)) where {P}
    nuisance = nuisance_parameters(ap)
    θhat = mode(ap)
    @inbounds for j in 1:i-1
        nuisance[j] = θhat[j]# + ap.map.cov[j,i] * sf
    end
    @inbounds for j in i:P-1
        nuisance[j] = θhat[j+1]# + ap.map.cov[j,i] * sf
    end
end

function profilepdf(ap::AbstractAsymptoticPosterior{P,T}, ::Val{reset_search_start}=Val{true}()) where {reset_search_start,P,T}
    i = profile_ind(ap)
    reset_search_start && naive_expected_nuisance!(ap, i)
    optimize_profile!(ap)
end
function profilepdf(ap::AbstractAsymptoticPosterior{P,T}, x, i, ::Val{reset_search_start}=Val{true}()) where {reset_search_start,P,T}
    set_profile_val_ind!(ap, x, i)
    # Reseset the other parameters to the corresponding mode.
    reset_search_start && naive_expected_nuisance!(ap, i)
    optimize_profile!(ap)
end

function (ap::AbstractAsymptoticPosterior)(theta, i = profile_ind(ap))
    debug() && println("")
    debug() && @show theta, rstar(ap), i
    debug() && @assert isfinite(theta)
    debug() && @show rstar_p(ap, theta, i) + rstar(ap)
    rstar_p(ap, theta, i) + rstar(ap)
end

# Positive if x is smaller than mode, negative if bigger
function rp(ap::AsymptoticPosteriorFD, x, i = profile_ind(ap))
    copysign( sqrt(2(delta_log_likelihood(ap))), mode(ap, i) - x)
end

function rstar_p(ap::AbstractAsymptoticPosterior, theta::Number, i::Int=profile_ind(ap))

    profilepdf(ap, theta, i)
    set_buffer_to_profile!(ap, i)

    setswap!(ap, i)
    hessian!(ap)

    debug() && @show ap.map.buffer
    debug() && @show hessian(ap)

    r = rp(ap, theta, i)
    q = qb(ap, theta, i)
#    @show r, q
    r + log(q/r)/r
end

sym(a,i) = Symbol(a, :_, i)
"""
Generates quote for vectorized multiplication of
    (L * g)' * (L * h)

"""
function profile_correction_quote(P, R, T)
    pad = R - P + 1
    # @show P,R
    VL = VectorizationBase.pick_vector_width(R, T)
    # VL = min(R, VectorizationBase.REGISTER_SIZE ÷ sizeof(T))
    VLT = VL * sizeof(T)
    V = Vec{VL,T}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote $(Expr(:meta, :inline)) end
    qa = q.args
    iter = R ÷ VL # number of iterations down the columns of the matrix.
    push!(qa, :(ptr_Li = pointer(Li)))
    push!(qa, :(@inbounds $(sym(:vH,0)) = SIMDPirates.vbroadcast($V,hess[1,$P]) ))
    push!(qa, :(@inbounds $(sym(:vG,0)) = SIMDPirates.vbroadcast($V,grad[1]) ))
    # r = 0
    push!(qa, :($(sym(:vL,0)) = vload($V, ptr_Li) ))
    push!(qa, :($(sym(:vHL,0)) = SIMDPirates.vmul( $(sym(:vH,0)), $(sym(:vL,0)) )))
    push!(qa, :($(sym(:vGL,0)) = SIMDPirates.vmul($(sym(:vG,0)), $(sym(:vL,0)) )))
    for r ∈ 1:iter-1 #ps = 0
        push!(qa, :($(sym(:vL,r)) = vload($V, ptr_Li + $(r*VLT)) ))
        push!(qa, :($(sym(:vHL,r)) = SIMDPirates.vmul($(sym(:vH,0)),$(sym(:vL,r)) )))
        push!(qa, :($(sym(:vGL,r)) = SIMDPirates.vmul($(sym(:vG,0)),$(sym(:vL,r)) )))
    end
    for ps ∈ 1:VL-pad-1
        push!(qa, :(@inbounds $(sym(:vH,ps)) = SIMDPirates.vbroadcast($V,hess[$(ps+1),$P])))
        push!(qa, :(@inbounds $(sym(:vG,ps)) = SIMDPirates.vbroadcast($V,grad[$(ps+1)])))
        for r ∈ 0:iter-1
            push!(qa, :($(sym(:vL,r)) = vload($V, ptr_Li + $(r*VLT + sizeof(T)*R*ps))))
            push!(qa, :($(sym(:vHL,r)) = SIMDPirates.vmuladd($(sym(:vH,ps)),$(sym(:vL,r)),$(sym(:vHL,r))) ))
            push!(qa, :($(sym(:vGL,r)) = SIMDPirates.vmuladd($(sym(:vG,ps)),$(sym(:vL,r)),$(sym(:vGL,r))) ))
        end
    end
    push!(qa, :(Vout = SIMDPirates.vmul($(sym(:vHL,0)), $(sym(:vGL,0)))))
    itermin = 1
    # for pb ∈ 0:VL:P-2
    #     for ps ∈ max(1,pb):min( pb+VL-1, P-2 )
    for pb ∈ VL-pad:VL:P-2
        for ps ∈ pb:min( pb+VL-1, P-2 )
            push!(qa, :(@inbounds $(sym(:vH,ps)) = SIMDPirates.vbroadcast($V,hess[$(ps+1),$P])))
            push!(qa, :(@inbounds $(sym(:vG,ps)) = SIMDPirates.vbroadcast($V,grad[$(ps+1)])))
            for r ∈ itermin:iter-1
                push!(qa, :($(sym(:vL,r)) = vload($V, ptr_Li + $(r*VLT + sizeof(T)*R*ps))))
                push!(qa, :($(sym(:vHL,r)) = SIMDPirates.vmuladd($(sym(:vH,ps)),$(sym(:vL,r)),$(sym(:vHL,r))) ))
                push!(qa, :($(sym(:vGL,r)) = SIMDPirates.vmuladd($(sym(:vG,ps)),$(sym(:vL,r)),$(sym(:vGL,r))) ))
            end
        end
        push!(qa, :(Vout = SIMDPirates.vmuladd($(sym(:vHL,itermin)), $(sym(:vGL,itermin)),Vout)))
        itermin += 1
    end
    # push!(qa, :(println("Finished profile correction.")))
    push!(qa, :(SIMDPirates.vsum(Vout)))
    q
end
"""
Calculates the quadratic form of grad' * Li' * Li * hess[1:end-1,end]
"""
@generated function profile_correction(Li::PaddedMatrices.AbstractMutableFixedSizeMatrix{R,Pm1,T,R},
                        grad::PaddedMatrices.AbstractFixedSizeVector, hess::PaddedMatrices.AbstractFixedSizeMatrix) where {Pm1,T,R}
    profile_correction_quote(Pm1+1, R, T)
end

function fdf_adjrstar_p(
    ap::AbstractAsymptoticPosterior{P,T}, theta, p_i::Int=profile_ind(ap),
    ::Val{reset_search_start} = Val{true}(), repeats = 0
) where {P,T,reset_search_start}

    profilepdf(ap, theta, p_i, Val{reset_search_start}())
    set_buffer_to_profile!(ap, p_i)
    setswap!(ap, p_i)
    hess = hessian!(ap)

    delta_log_like = delta_log_likelihood(ap)
    @inbounds r = copysign( sqrt((T(2))*delta_log_like), mode(ap, p_i) - theta)
    # r = rp(ap, theta, p_i)

    grad = gradient(ap)

    Li = profile_lower_triangle(ap)
    rootdet = PaddedMatrices.invcholdetLLc!(Li, hess)

    prof_factor = profile_correction(Li, grad, hess)
    hess_adjust = rootdet * base_adjustment(ap)
    @inbounds q = (prof_factor - grad[P]) * hess_adjust
    #    @show r, q, delta_log_like, rootdet, base_adjustment(ap), hess_adjust
    if sign(q)  != sign(r)
        repeats > 10 && return T(-Inf)
        refit!(ap)
        return fdf_adjrstar_p(ap, theta, p_i, Val{reset_search_start}(), repeats + 1)
    end
    r⭐ = r + log(q/r)/r
#    r⭐ + rstar(ap), exp(T(0.5)*abs2(r⭐)-delta_log_like) / hess_adjust#, exp(-delta_log_like) / hess_adjust
    1 - Φ(r⭐) - α(ap), exp(-delta_log_like) / (Base.FastMath.sqrt_fast(2π) * rootdet * base_adjustment(ap))
#    Φ(r⭐ + rstar(ap)), exp(delta_log_like) / (Base.FastMath.sqrt_fast(2π) * rootdet * base_adjustment(ap))
end

function cdf(
    ap::AbstractAsymptoticPosterior{P,T}, theta, p_i::Int=profile_ind(ap),
    ::Val{reset_search_start} = Val{true}(), tries = 0
) where {P,T,reset_search_start}

    profilepdf(ap, theta, p_i, Val{reset_search_start}())
    set_buffer_to_profile!(ap, p_i)
    setswap!(ap, p_i)
    hess = hessian!(ap)
    delta_log_like = delta_log_likelihood(ap)
    @inbounds r = copysign( sqrt((T(2))*delta_log_like), mode(ap, p_i) - theta)
    grad = gradient(ap)
    Li = profile_lower_triangle(ap)
    rootdet = PaddedMatrices.invcholdetLLc!(Li, hess)
    prof_factor = profile_correction(Li, grad, hess)
    hess_adjust = rootdet * base_adjustment(ap)
    @inbounds q = (prof_factor - grad[P]) * hess_adjust
    if sign(q) != sign(r)
        refit!(ap)
        if tries > 10
            @warn "Tried refitting 10 times."
            return -Inf
        end
        return cdf(ap, theta, p_i, Val{reset_search_start}(), tries + 1)
    end
    r⭐ = r + log(q/r)/r

    1 - Φ(r⭐)
end
@generated function subhessian(H::AbstractMutableFixedSizeMatrix{P,P,T,R}) where {P,T,R}
    Pm1 = P - 1
    quote
        $(Expr(:meta,:inline))
        PtrMatrix{$Pm1,$Pm1,$T,$R,$(Pm1*R)}(pointer(H))
    end
end

function pdf(ap::AbstractAsymptoticPosterior{P,T,Pm1}, theta, i = profile_ind(ap),::Val{reset_search_start}=Val{true}()) where {P,T,Pm1,reset_search_start}

    profilepdf(ap, theta, i, Val{reset_search_start}())
    set_buffer_to_profile!(ap, i)
    setswap!(ap, i)
    hessian!(ap.map.od.config, ap.map.buffer)

    # is there a more efficient way of calculating the determinant?
    # skip storing? Implement that...
    # rootdet = PaddedMatrices.choldet!(hessian(ap), Val{Pm1}())
    # isfinite(rootdet) ? exp(delta_log_likelihood(ap)) / (sqrt(2π) * rootdet * base_adjustment(ap)) : T(Inf)
    rootdet, success = PaddedMatrices.safecholdet!(subhessian(hessian(ap)))
    success ? exp(-delta_log_likelihood(ap)) / (sqrt(2π) * rootdet * base_adjustment(ap)) : T(-Inf)
end


function qb(ap::AbstractAsymptoticPosterior{P}, theta, pi = profile_ind(ap)) where {P}
    grad = gradient(ap)
    hess = hessian(ap) # P x P
    Li = profile_lower_triangle(ap)
    rootdet = PaddedMatrices.invcholdetLLc!(Li, hess)
    prof_factor = profile_correction(Li, grad, hess)

    @inbounds (prof_factor - grad[P]) * rootdet * base_adjustment(ap)
end

function set_buffer_to_profile!(ap::AbstractAsymptoticPosterior{P}, i = profile_ind(ap)) where {P}
    nuisance = nuisance_parameters(ap)
    buffer = initial_val_buffer(ap)
    @inbounds for j in 1:i-1
        buffer[j] = nuisance[j]
    end
    if i != P
        @inbounds buffer[i] = nuisance[P-1]
        @inbounds for j in i+1:P-1
            buffer[j] = nuisance[j-1]
        end
    end
    buffer[P] = profile_val(ap)
    nothing
end

Φ(x::T) where {T} = muladd(T(0.5), erf( x / Base.FastMath.sqrt_fast(2) ), T(0.5))
Φ⁻¹(x::T) where {T} = Base.FastMath.sqrt_fast(T(2))*erfinv(muladd(T(2),x,T(-1)))
function Statistics.quantile(ap::AbstractAsymptoticPosterior, alpha, i)
    profile_ind!(ap, i)
    α!(ap, alpha)
#    rstar!(ap, Φ⁻¹(alpha))
    # quadratic_search(ap, i)
    linear_search(ap, profile_ind(ap))
end
function Statistics.quantile(ap::AbstractAsymptoticPosterior, alpha)
    α!(ap, alpha)
#    rstar!(ap, Φ⁻¹(alpha))
    linear_search(ap, profile_ind(ap))
    # quadratic_search(ap, profile_ind(ap))
end


export lquantile, qquantile
function lquantile(ap::AbstractAsymptoticPosterior, alpha, i)
    profile_ind!(ap, i)
    α!(ap, alpha)
    linear_search(ap, i)
end
function lquantile(ap::AbstractAsymptoticPosterior, alpha)
    α!(ap, alpha)
#    rstar!(ap, Φ⁻¹(alpha))
    linear_search(ap, profile_ind(ap))
end

function qquantile(ap::AbstractAsymptoticPosterior, alpha, i)
    profile_ind!(ap, i)
    α!(ap, alpha)
#    rstar!(ap, Φ⁻¹(alpha))
    quadratic_search(ap, i)
end
function qquantile(ap::AbstractAsymptoticPosterior, alpha)
    α!(ap, alpha)
#    rstar!(ap, Φ⁻¹(alpha))
    quadratic_search(ap, profile_ind(ap))
end


# (ap::AsymptoticPosteriorFD)(x) = ap.map.od(x)



