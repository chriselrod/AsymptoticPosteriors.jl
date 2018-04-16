

"""Here, we take a much more straightforward approach to the implementation of Reid."""

mutable struct Swap{N,F}
    f::F
    i::Int
end

struct AsymptoticMAP{N,T,C,D,M,O,S}
    configuration::C
    od::D
    method::M
    options::O
    state::S
    θhat::Vector{T}
    lmax::RefValue{T}
    base_adjust::RefValue{T}
    lastval::RefValue{T}
    last_x::Vector{T}
    std_estimates::Vector{T}
    cov_mat::Matrix{T}
    quantile::RefValue{Float64}
end

struct ProfileLikelihood{L}
    ℓ::L
    x::Vector{T}

end

struct ReidPosterior



end
struct HessianConfig{T,V,N,DG,DJ,DR} <: ForwardDiff.AbstractConfig{N}
    jacobian_config::ForwardDiff.JacobianConfig{T,V,N,DJ}
    gradient_config::ForwardDiff.GradientConfig{T,Dual{T,V,N},N,DG}
    inner_result::DiffResults.MutableDiffResult{Dual{T,V,N},Vector{Dual{T,V,N}}}
end
function HessianConfig(f::F, result::DiffResults.DiffResult, x::AbstractArray{V}, chunk::Chunk = Chunk(x), tag = Tag(f, V)) where {F,V}
    jacobian_config = ForwardDiff.JacobianConfig((f,gradient), DiffResults.gradient(result), x, chunk, tag)
    gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    inner_result = DiffResults.DiffResult(zero(eltype(jacobian_config.duals[2])), jacobian_config.duals[2])
    return HessianConfig(jacobian_config, gradient_config, inner_result)
end
struct GradF{R,CFG,F}
    result::R
    cfg::CFG
    f::F
end
function (x::GradF)(y, z)
    x.cfg.inner_result.derivs = (y,) #Already true?
    ForwardDiff.gradient!(x.cfg.inner_result, x.f, z, x.cfg.gradient_config, Val{false}())
    x.result = DiffResults.value!(x.result, ForwardDiff.value(DiffResults.value(x.cfg.inner_result)))
    return y
end
function hessian!(result::DiffResults.DiffResult, f, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, result, x), ::Val{CHK}=Val{true}()) where {T,CHK}
    CHK && ForwardDiff.checktag(T, f, x)
    hessian!(GradF(result, cfg, f), x)
    return result
end
function hessian!(∇f!::GradF, x::AbstractArray) where {T,CHK}
    ForwardDiff.jacobian!(DiffResults.hessian(∇f!.result), ∇f!, DiffResults.gradient(∇f!.result), x, ∇f!.cfg.jacobian_config, Val{false}())
    return ∇f!
end
struct DF{C}
    config::C
end
(df::DF)(out, x) = ForwardDiff.gradient!(out, df.config.swap.f, x, df.config.gconfig, Val{false}())
struct FDF{C}
    config::C
end
function (fdf::FDF)(out, x)
    fdf.config.result.derivs = (out,fdf.config.result.derivs[2])
    ForwardDiff.gradient!(fdf.config.result, fdf.config.swap.f, x, fdf.config.gconfig, Val{false}())
    DiffResults.value(fdf.config.result)
end

function AsymptoticMAP(initial_x::AbstractVector{T}, config, ::Val{N}) where {T,N}
    df = DF(config)
    fdf = FDF(config)

    odmax = OnceDifferentiable(config.swap.f, df, fdf, initial_x, zero(T), Val{true}())
    # LBFGS(linesearch=LineSearches.BackTracking())

    backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), x -> Matrix{eltype(x)}(I, length(x), length(x)), Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)

    options = Optim.InternalUseOptions(backtrack)
    state = Optim.initial_state(backtrack, options, odmax, initial_x)

    opt = optimize(odmax, initial_x, backtrack, options, state)
    θhat = Optim.minimizer(opt)

    hessian!(config.∇f!, θhat)

    cov_mat = copy(DiffResults.hessian(config.result))


    root_cov_det = choldet!(cov_mat, UpperTriangular, Val{N}())
    Compat.LinearAlgebra.LAPACK.potri!('U', cov_mat) #Computes inverse from chol
    std_estimates = Vector{T}(undef, N)
    for i ∈ 1:N
        for j ∈ 1:i-1
            cov_mat[i,j] = cov_mat[j,i]
        end
        std_estimates[i] = sqrt(cov_mat[i,i])
    end
    lmax = Optim.minimum(opt)
    base_adjust = 1 / root_cov_det  


    AsymptoticMAP(config, odmax, backtrack, options, state, θhat, Ref(lmax), Ref(base_adjust), Ref{T}(), Vector{T}(undef, N+1), std_estimates, cov_mat, Ref{Float64}(), Val{N}())
end
function AsymptoticMAP(configuration::C, od::D, method::M, options::O, state::S, θhat::Vector{T}, lmax::Ref{T}, base_adjust::Ref{T}, lastval::Ref{T}, last_x::Vector{T}, std_estimates::Vector{T}, cov_mat::Matrix{T}, quantile::Ref{Float64}, ::Val{N}) where {N,T,C,D,M,O,S}
    AsymptoticMAP{N,T,C,D,M,O,S}(configuration, od, method, options, state, θhat, lmax, base_adjust, lastval, last_x, std_estimates, cov_mat, quantile)
end


function AsymptoticConfig(l::L, initial_x::AbstractVector{T}, ::Val{N}) where {N,T,L}
    chunk = ChunkNotPirate(Val{N}())
    result = DiffResults.HessianResult(initial_x)

    gconfig = ForwardDiff.GradientConfig(nothing, initial_x, chunk)
    swap = Swap{N,L}(l,N)
    ∇f! = GradF(result, ForwardDiff.HessianConfig(nothing, result, initial_x, chunk), swap )

    AsymptoticConfig(result, gconfig, ∇f!, swap)
end

function (s::Swap{N})(x) where N
    @inbounds x[s.i], x[N] = x[N], x[s.i]
    s.f(x)
end
