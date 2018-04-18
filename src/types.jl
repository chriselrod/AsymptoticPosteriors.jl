# Chicken and egg problem...or, disable tag checks, and make sure
# that you have different tags.
struct AsymptoticConfig{T,R,GC,HC,FC,L}
    result::R
    gresult::Vector{T}
    gconfig::GC
    hconfig::HC
    finite_cache::FC
    swap::RefValue{Int}
    l::L
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
    fo_std_estimates::Vector{T}
    cov_mat::Matrix{T}
    quantile::RefValue{Float64}
end

struct AsymptoticPosterior{N,T,MAP<:AsymptoticMAP{N,T},D,M,O,S}
    map::MAP
    od::D
    method::M
    options::O
    state::S
end

@generated ChunkNotPirate(::Val{N}) where N = ForwardDiff.Chunk(N)
@generated ChunkNotPirateP1(::Val{N}) where N = ForwardDiff.Chunk(N+1)

const HasTypedLength{N,T} = Union{NTuple{N,T}, SVector{N,T}}
vallength(::HasTypedLength{N}) where N = Val{N}()
vallength(x) = Val{length(x)}() #not type stable; ensure function barrier exists?

# Do we type on L?
# Aim of this function:
# (1) Set up everything needed; changing input data?
#     (a) function


# Add @noinline to guarantee function barrier?
AsymptoticPosterior(l, initial_x) = AsymptoticPosterior(l, initial_x, vallength(initial_x))
function AsymptoticPosterior(l::L, initial_x::AbstractVector{T}, ::Val{N}) where {L,T,N}
    config = AsymptoticConfig(l, initial_x, Val{N}())
    map = AsymptoticMAP(initial_x, config, Val{N}())
    AsymptoticPosterior(map)
end

function arrays_equal(x, y)
    @boundscheck if length(x) != length(y)
        return false
    end
    all_equal = true
    @inbounds for i ∈ eachindex(x)
        if x[i] != y[i]
            all_equal = false
            break
        end
    end
    return all_equal
end


function AsymptoticMAP(initial_x::AbstractVector{T}, config, ::Val{N}) where {T,N}
    df = (out, x) -> ForwardDiff.gradient!(out, config.l, x, config.gconfig, Val{false}())
    fdf = (out, x) -> begin
        config.result.derivs = (out,config.result.derivs[2])
        ForwardDiff.gradient!(config.result, config.l, x, config.gconfig, Val{false}())
        DiffResults.value(config.result)
    end

    odmax = OnceDifferentiable(config.l, df, fdf, initial_x, zero(T), Val{true}())
    # LBFGS(linesearch=LineSearches.BackTracking())

    backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), x -> Matrix{eltype(x)}(I, length(x), length(x)), Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)

    options = Optim.InternalUseOptions(backtrack)
    state = Optim.initial_state(backtrack, options, odmax, initial_x)


    opt = optimize(odmax, initial_x, backtrack, options, state)
    θhat = Optim.minimizer(opt)
    ForwardDiff.hessian!(config.result, config.l, θhat, config.hconfig, Val{false}())

    cov_mat = copy(DiffResults.hessian(config.result))
    herm_det, prof_info, unused = profile_hessian!(DiffResults.hessian(config.result), Val{N}())

    Compat.LinearAlgebra.LAPACK.potrf!('U', cov_mat)
    Compat.LinearAlgebra.LAPACK.potri!('U', cov_mat)
    fo_std_estimates = Vector{T}(undef, N)
    for i ∈ 1:N
        for j ∈ 1:i-1
            cov_mat[i,j] = cov_mat[j,i]
        end
        fo_std_estimates[i] = sqrt(cov_mat[i,i])
    end
    lmax = Optim.minimum(opt)


    base_adjust = 0
    base_adjust = 0


    AsymptoticMAP(config, odmax, backtrack, options, state, θhat, Ref(lmax), Ref(base_adjust), Ref{T}(), Vector{T}(undef, N+1), fo_std_estimates, cov_mat, Ref{Float64}(), Val{N}())
end
function AsymptoticMAP(configuration::C, od::D, method::M, options::O, state::S, θhat::Vector{T}, lmax::Ref{T}, base_adjust::Ref{T}, lastval::Ref{T}, last_x::Vector{T}, fo_std_estimates::Vector{T}, cov_mat::Matrix{T}, quantile::Ref{Float64}, ::Val{N}) where {N,T,C,D,M,O,S}
    AsymptoticMAP{N,T,C,D,M,O,S}(configuration, od, method, options, state, θhat, lmax, base_adjust, lastval, last_x, fo_std_estimates, cov_mat, quantile)
end

# Okay, now I have to figure out creation of once differenitable object and function for solving
# the lagrange multiplier problem to find bounds.
# Also, be smart about about defining intervals.


function reset_optim_state!(map, initial_x)
    copyto!(map.state.x, initial_x)
    Optim.retract!(map.method.manifold, map.state.x)
    Optim.value_gradient!!(map.od, map.state.x)
    Optim.project_tangent!(map.method.manifold, NLSolversBase.gradient(map.od), map.state.x)
end

function AsymptoticConfig(l::L, initial_x::AbstractVector{T}, ::Val{N}) where {N,T,L}
    chunk = ChunkNotPirate(Val{N}())
    result = DiffResults.HessianResult(initial_x)

    gresult = Vector{T}(undef, N+1)
    finite_cache = DiffEqDiffTools.DiffEqDiffTools.GradientCache{Nothing,Nothing,Nothing,:central,T,true}(nothing, nothing, nothing)#Val{:forward}

    gconfig = ForwardDiff.GradientConfig(l, initial_x, chunk)
    hconfig = ForwardDiff.HessianConfig(nothing, result, initial_x, chunk)

    AsymptoticConfig(result, gresult, gconfig, hconfig, finite_cache, Ref(N), l)
end

# To pull off ind swapping, need to swap
# wrap log likelihood, swap its inputs
# the initial values passed to it
# swap:
# params start in order 1,2,3,4,5; i = 3 -> 1,2,5,4,3

# Need to set qb_constant

function update_grad!(map, x)
    copyto!(map.last_x, x)
    DiffEqDiffTools.finite_difference_gradient!(map.configuration.gresult, y->LagrangeGrad(map,y), x, map.configuration.finite_cache)
    map.lastval[] = sum(abs2, map.configuration.gresult)/2
    # sum(abs2, map.configuration.gresult)/2
    # map.lastval[]
    # lv
end
function update_grad!(out, map, x)
    arrays_equal(x, map.last_x) || update_grad!(map, x)
    copyto!(out, map.configuration.gresult)
    nothing
end


function AsymptoticPosterior(map::M) where {T, N, M <: AsymptoticMAP{N,T}}
    F = x -> begin
        arrays_equal(x, map.last_x) ? map.lastval[] : update_grad!(map, x)
    end
    G = (out, x) -> begin
        update_grad!(out, map, x)
        out
    end
    FG = (out, x) -> begin
        update_grad!(out, map, x)
        map.lastval[]
    end

    x = Vector{T}(undef, N+1)
    @inbounds begin
        for i ∈ 1:N
            x[i] = map.θhat[i]
        end
        x[end] = zero(T)
    end

    odprof = OnceDifferentiable(F, G, FG, x, zero(T), Val{true}())
    backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), y -> Matrix{T}(I, length(y), length(y)), Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)
    options = Optim.InternalUseOptions(backtrack)
    state = Optim.initial_state(backtrack, options, odprof, x)

    AsymptoticPosterior(map, odprof, backtrack, options, state, Val{N}())
end

function AsymptoticPosterior(map::MAP, od::D, method::M, options::O, state::S, ::Val{N}) where {N,T,MAP<:AsymptoticMAP{N,T},D,M,O,S}
    AsymptoticPosterior{N,T,MAP,D,M,O,S}(map, od, method, options, state)
end

function swap!(x, i, n)
    @inbounds x[i], x[n] = x[n], x[i]
    x
end

# The Lagrange Function, whose gradient we're zeroing.
function LagrangeGrad(map::AsymptoticMAP{N}, x) where N
    theta = @view(x[1:N])
    @show x
    update_hessian!(map, theta)
    (rstar_p(map, theta)-map.quantile[]) * x[end] - DiffResults.value(map.configuration.result)
end

function update_hessian!(map::AsymptoticMAP{N}, theta) where N
    ForwardDiff.hessian!(map.configuration.result, x -> map.configuration.l(swap!(x, map.configuration.swap[], N)), theta, map.configuration.hconfig, Val{false}())
    @show DiffResults.hessian(map.configuration.result)
end


function bound(posterior::AsymptoticPosterior{N}, param::Integer, alpha) where N
    @boundscheck @assert param <= N
    posterior.map.quantile[] = Φ⁻¹(alpha)
    posterior.map.configuration.swap[] = param
    copyto!(posterior.state.x, posterior.map.θhat)
    δ = posterior.map.quantile[] * posterior.map.fo_std_estimates[param]
    posterior.state.x[param] = posterior.state.x[param] + δ
    swap!(posterior.state.x, param, N)
    swap!(posterior.state.invH, posterior.map.cov_mat, param, N)

    Σ11 = posterior.state.invH[1:N-1,1:N-1]
    Compat.LinearAlgebra.LAPACK.potrf!('U', Σ11)
    Compat.LinearAlgebra.LAPACK.potri!('U', Σ11)
    BLAS.symv!('U', δ, Σ11, @view(posterior.state.invH[1:N-1,N]), 1.0, @view(posterior.state.x[1:7]))

    Optim.retract!(posterior.method.manifold, posterior.state.x)
    Optim.value_gradient!!(posterior.od, posterior.state.x)
    Optim.project_tangent!(posterior.method.manifold, Optim.gradient(posterior.od), posterior.state.x)
    @show posterior.state.x
    optimize(posterior.od, posterior.state.x, posterior.method, posterior.options, posterior.state)
end

"""
Copies leading n x n principal submatrix of B into A, but swaps contents of row and column k and n, where k < n.

"""
function swap!(A::Matrix{T}, B::Matrix{T}, k::Int, n::Int) where T
    @inbounds begin
        for i ∈ 1:k-1
            for j ∈ 1:k-1
                A[j,i] = B[j,i]
            end
            A[n,i] = B[k,i]
            for j ∈ k+1:n-1
                A[j,i] = B[j,i]
            end
            A[k,i] = B[n,i]
        end
        for j ∈ 1:k-1
            A[j,n] = B[j,k]
        end
        A[n,n] = B[k,k]
        for j ∈ k+1:n-1
            A[j,n] = B[j,k]
        end
        A[k,n] = B[n,k]
        for i ∈ k+1:n-1
            for j ∈ 1:k-1
                A[j,i] = B[j,i]
            end
            A[n,i] = B[k,i]
            for j ∈ k+1:n-1
                A[j,i] = B[j,i]
            end
            A[k,i] = B[n,i]
        end
        for j ∈ 1:k-1
            A[j,k] = B[j,n]
        end
        A[n,k] = B[k,n]
        for j ∈ k+1:n-1
            A[j,k] = B[j,n]
        end
        A[k,k] = B[n,n]
    end
    nothing
end

function set_I!(x::AbstractMatrix{T}, ::Val{N}) where {N,T}
    @inbounds for i ∈ 1:N
        for j ∈ 1:i-1
            x[j,i] = zero(T)
        end
        x[i,i] = one(T)#/10^4 10^4 is a little overzealous, isn't it?
        for j ∈ i+1:N
            x[j,i] = zero(T)
        end
    end
end

function rp(map, theta)
    sqrt(2*(DiffResults.value(map.configuration.result)-map.lmax[]))
end

# This is what is set to phi-inv
function rstar_p(map, theta)
    r = rp(map, theta)
    sign(map.θhat[map.configuration.swap[]]-theta[end])*(r + log(abs(qb(map, theta)/r))/r)
end

# function rp(map, theta)
#     sign(map.θhat[map.configuration.swap[]]-theta[end])*sqrt(2*(DiffResults.value(map.configuration.result)-map.lmax[]))
# end

# # This is what is set to phi-inv
# function rstar_p(map, theta)
#     r = rp(map, theta)
#     r + log(qb(map, theta)/r)/r
# end

function qb(map::AsymptoticMAP{N}, theta) where N
    prof_score = DiffResults.gradient(map.configuration.result)[end]
    root_herm_det, prof_info, prof_info_uncorrected = profile_hessian!(DiffResults.hessian(map.configuration.result), Val{N}()) # destroys hessian
    root_herm_det * map.base_adjust[] * prof_score * prof_info / prof_info_uncorrected
end

function hessian!(posterior::AsymptoticPosterior, theta)
    ForwardDiff.hessian!(posterior.results, posterior.l, theta, posterior.hess_config)
end


# function _profile_hessian!(N)
#     Nm1 = N-1
#     quote
#         # hess = DiffResults.hessian(posterior.Dresult)
#         @fastmath begin
#             # Test having this be a view instead
#             Base.Cartesian.@nextract $Nm1 h i -> hess[i,$N]
#             #Profiled information.
#             prof_info = prof_info_uncorrected = hess[end]

#             # Calculate Lambda_{12}*Lambda_{22}^{-1}*Lambda_{21}
#             # Quad form; calc via Lamda_{22} |> triangle |> inv triangle |>
#             #     triangle*vector |> vec dot self
#             root_herm_det = choldet!(hess, UpperTriangular, Val{$Nm1}())# calc det in here
#             inv!(hess, Val{$Nm1}(), UpperTriangular)
#             Base.Cartesian.@nexprs $Nm1 i -> begin
#                 temp = zero(eltype(hess))
#                 Base.Cartesian.@nexprs i j -> begin
#                     temp += h_j * hess[j,i]
#                 end
#                 prof_info -= abs2(temp)
#             end
#         end
#         root_herm_det, prof_info, prof_info_uncorrected
#     end
# end


# Swap param orders so Hessian is 1:N-1,1:N-1, and POI in last block?
# Seems like that one change will be more efficient than mucking around with this nonsense
# @generated function profile_hessian!(hess, ::Val{D}) where D
#     _profile_hessian!(D)
# end


@generated Valp1(::Val{N}) where N = Val{N+1}()
@generated Valm1(::Val{N}) where N = Val{N-1}()