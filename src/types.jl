# Chicken and egg problem...or, disable tag checks, and make sure
# that you have different tags.

struct AsymptoticConfig{R,GR,HR,GC,HC,DGC,DHC,V}
    result::R
    gresult::GR
    hresult::HR
    gconfig::GC
    hconfig::HC
    Dgconfig::DGC
    Dhconfig::DHC
    theta::V# a veiw
end
struct AsymptoticMAP{N,T,L,C,D,M,O,S}
    l::L
    configuration::C
    od::D
    method::M
    options::O
    state::S
    θhat::Vector{T}
    lmax::Ref{T}
    base_adjust::Ref{T}
    lastval::Ref{T}
    last_x::Vector{T}
    quantile::Ref{T}
    swap::Ref{Int}
end

struct AsymptoticPosterior{N,T,MAP<:AsymptoticMAP{N,T},D,M,O,S}
    map::MAP
    initial_x::Vector{T}
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
    map = AsymptoticMAP(l, initial_x, config, Val{N}())
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


function AsymptoticMAP(l::L, initial_x::AbstractVector{T}, config, ::Val{N}) where {L,T,N}
    df = (out, x) -> ForwardDiff.gradient!(out, l, x, config.gconfig, Val{false}())
    fdf = (out, x) -> begin
        config.result.derivs = (out,config.result.derivs[2])
        ForwardDiff.gradient!(config.result, l, x, config.gconfig, Val{false}())
        DiffResults.value(config.result)
    end

    odmax = OnceDifferentiable(l, df, fdf, initial_x, zero(T), Val{true}())
    # LBFGS(linesearch=LineSearches.BackTracking())

    backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), x -> Matrix{eltype(x)}(I, length(x), length(x)), Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)

    options = Optim.InternalUseOptions(backtrack)
    state = Optim.initial_state(backtrack, options, odmax, initial_x)

    opt = optimize(odmax, initial_x, backtrack, options, state)
    θhat = Optim.minimizer(opt)
    ForwardDiff.hessian!(config.result, l, θhat, config.hconfig, Val{false}())
    herm_det, prof_info, unused = profile_hessian!(DiffResults.hessian(config.result), Val{N}())
    lmax = Optim.minimum(opt)
    base_adjust = inv(herm_det*sqrt(prof_info))
    AsymptoticMAP(l, config, odmax, backtrack, options, state, θhat, Ref(lmax), Ref(base_adjust), Ref{T}(), Vector{T}(undef, N+1), Ref{Float64}(), Ref(N), Val{N}())
end
function AsymptoticMAP(l::L, configuration::C, od::D, method::M, options::O, state::S, θhat::Vector{T}, lmax::Ref{T}, base_adjust::Ref{T}, lastval::Ref{T}, last_x::Vector{T}, quantile::Ref{T}, swap::Ref{Int}, ::Val{N}) where {N,T,L,C,D,M,O,S}
    AsymptoticMAP{N,T,L,C,D,M,O,S}(l, configuration, od, method, options, state, θhat, lmax, base_adjust, lastval, last_x, quantile, swap)
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
    chunkp1 = ChunkNotPirateP1(Val{N}())
    result = DiffResults.HessianResult(initial_x)
    gconfig = ForwardDiff.GradientConfig(l, initial_x, chunk)
    hconfig = ForwardDiff.HessianConfig(l, result, initial_x, chunk)

    gresult = DiffResults.MutableDiffResult(zero(T), (Vector{T}(undef,N+1),))
    Dgconfig = ForwardDiff.GradientConfig(nothing, gresult.derivs[1], chunkp1)

    theta = @view(Dgconfig.duals[1:N])
    hresult = DiffResults.HessianResult(theta)
    Dhconfig = ForwardDiff.HessianConfig(nothing, hresult, theta, chunk)

    AsymptoticConfig(result, gresult, hresult, gconfig, hconfig, Dgconfig, Dhconfig, theta)
end

# To pull off ind swapping, need to swap
# wrap log likelihood, swap its inputs
# the initial values passed to it
# swap:
# params start in order 1,2,3,4,5; i = 3 -> 1,2,5,4,3

# Need to set qb_constant

function update_grad!(map, x)
    copyto!(map.last_x, x)
    ForwardDiff.gradient!(map.configuration.gresult, y->LagrangeGrad(map,y), x, map.configuration.Dgconfig)
    lv = sum(abs2, DiffResults.gradient(map.configuration.gresult))/2
    map.lastval[] = lv
    lv
end

function AsymptoticPosterior(map::AsymptoticMAP{N,T}) where {T, N}
    F = x -> begin
        arrays_equal(x, map.last_x) ? map.lastval[] : update_grad!(map, x)
    end
    G = (out, x) -> begin
        if arrays_equal(x, map.last_x)
            grad = DiffResults.gradient(map.configuration.gresult)
            out === grad || copyto!(out, grad) #Does this make sense? Could the two objects end up being one and the same?
        else
            map.configuration.gresult.derivs = (out,)
            update_grad!(map, x)
        end
        out
    end
    FG = (out, x) -> begin
        if arrays_equal(x, map.last_x)
            grad = DiffResults.gradient(map.configuration.gresult)
            out === grad || copyto!(out, grad) #Does this make sense? Could the two objects end up being one and the same?
        else
            map.configuration.gresult.derivs = (out,)
            update_grad!(map, x)
        end
        map.lastval[]
    end

    initial_x = Vector{T}(undef, N+1)
    @inbounds begin
        for i ∈ 1:N
            initial_x[i] = map.θhat[i]
        end
        initial_x[end] = zero(T)
    end

    odmax = OnceDifferentiable(F, G, FG, initial_x, zero(T), Val{true}())
    backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), x -> Matrix{eltype(x)}(I, length(x), length(x)), Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)
    options = Optim.InternalUseOptions(backtrack)
    state = Optim.initial_state(backtrack, options, odmax, initial_x)

    AsymptoticPosterior(map, initial_x, odmax, backtrack, options, state, Val{N}())
end

function AsymptoticPosterior(map::MAP, initial_x::Vector{T}, od::D, method::M, options::O, state::S, ::Val{N}) where {N,T,MAP<:AsymptoticMAP{N,T},D,M,O,S}
    AsymptoticPosterior{N,T,MAP,D,M,O,S}(map, initial_x, od, method, options, state)
end

function swap!(x, i, n)
    @inbounds x[i], x[n] = x[n], x[i]
    x
end

# The Lagrange Function, whose gradient we're zeroing.
function LagrangeGrad(map::AsymptoticMAP{N}, x) where N
    @show x
    @show typeof(x)
    @show map.configuration.theta
    @show typeof(map.configuration.theta)
    update_hessian!(map)
    drv = DiffResults.value(map.configuration.hresult)
    diff = (rstar_p(map,map.configuration.theta)-map.quantile[])
    @show diff
    @show typeof(diff)
    @show x[end]
    @show typeof(x[end])
    dprod = diff * x[end]
    dprod - drv
    # (rstar_p(map,map.configuration.theta)-map.quantile[]) * x[end] - DiffResults.value(map.configuration.hresult)
end

function update_hessian!(map::AsymptoticMAP{N}) where N
    ForwardDiff.hessian!(map.configuration.hresult, y -> map.l(swap!(y, map.swap[], N)), map.configuration.theta, map.configuration.Dhconfig, Val{false}())
    nothing
end

Φ⁻¹(x) = √2*erfinv(2x-1)

function bound(posterior::AsymptoticPosterior{N}, param::Integer, alpha) where N
    @boundscheck @assert param <= N
    posterior.quantile[] = Φ⁻¹(alpha)
    posterior.swap[] = param
    optimize(posterior, alpha)
end

function rp(map, theta)
    sign(map.θhat[map.swap[]]-theta[end]) * sqrt(2*(map.lmax[]-DiffResults.value(map.configuration.hresult)))
end

# This is what is set to phi-inv
function rstar_p(map, theta)
    r = rp(map, theta)
    r + log(qb(map, theta)/r)/r
end

function qb(map::AsymptoticMAP{N}, theta) where N
    prof_score = DiffResults.gradient(map.configuration.hresult)[end]
    root_herm_det, prof_info, prof_info_uncorrected = profile_hessian!(DiffResults.hessian(map.configuration.hresult), Val{N}()) # destroys hessian
    root_herm_det * map.base_adjust[] * prof_score * prof_info / prof_info_uncorrected
end

function hessian!(posterior::AsymptoticPosterior, theta)
    ForwardDiff.hessian!(posterior.results, posterior.l, theta, posterior.hess_config)
end


function _profile_hessian!(N)
    Nm1 = N-1
    quote
        # hess = DiffResults.hessian(posterior.Dresult)
        @fastmath begin
            # Test having this be a view instead
            Base.Cartesian.@nextract $Nm1 h i -> hess[i,$N]
            #Profiled information.
            prof_info = prof_info_uncorrected = hess[end]

            # Calculate Lambda_{12}*Lambda_{22}^{-1}*Lambda_{21}
            # Quad form; calc via Lamda_{22} |> triangle |> inv triangle |>
            #     triangle*vector |> vec dot self
            root_herm_det = choldet!(hess, UpperTriangular, Val{$Nm1}())# calc det in here
            inv!(hess, Val{$Nm1}(), UpperTriangular)
            Base.Cartesian.@nexprs $Nm1 i -> begin
                temp = zero(eltype(hess))
                Base.Cartesian.@nexprs i j -> begin
                    temp += h_j * hess[j,i]
                end
                prof_info -= abs2(temp)
            end
        end
        root_herm_det, prof_info, prof_info_uncorrected
    end
end


# Swap param orders so Hessian is 1:N-1,1:N-1, and POI in last block?
# Seems like that one change will be more efficient than mucking around with this nonsense
@generated function profile_hessian!(hess, ::Val{D}) where D
    _profile_hessian!(D)
end

# More reliable than Base.@pure ???
@generated Valm(::Val{N}) where N = Val{N-1}()
# LinearAlgebra.LAPACK.potrf!('U', x) #Only good for LinearAlgebra.BLAS.BlasFloat
# LinearAlgebra.LAPACK.trtri!('U', 'N', x)


