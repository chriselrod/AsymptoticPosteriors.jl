

"""Here, we take a much more straightforward approach to the implementation of Reid."""

struct ProfileWrapper{N,F,T,A<:AbstractArray{T}}
    f::F
    x::A
    i::Base.RefValue{Int}
    v::Base.RefValue{T}
end
function (pw::ProfileWrapper{N})(x) where N
    @inbounds begin
        for i in 1:pw.i[]-1
            pw.x[i] = x[i]
        end
        # pw.x[pw.i[]] = pw.v[]
        for i in pw.i[]+1:N
            pw.x[i] = x[i-1]
        end
    end
    - pw.f(pw.x)
end
function set!(pw::ProfileWrapper, v, i::Int)
    pw.v[] = v
    pw.i[] = i
    @inbounds pw.x[i] = v
    pw
end

struct Swap{N,F}
    f::F
    i::Base.RefValue{Int}
end
function (s::Swap{N})(x) where N
    @inbounds x[s.i[]], x[N] = x[N], x[s.i[]]
    - s.f(x)
end



struct MAP{N,T,D,M,O,S}
    od::D
    method::M
    options::O
    state::S
    θhat::Vector{T}
    buffer::Vector{T}
    nlmax::Base.RefValue{T}
    base_adjust::Base.RefValue{T}
    std_estimates::Vector{T}
    cov::Matrix{T}
end

struct ProfileLikelihood{N,T,D,M,S,MAP_ <: MAP{N,T},A<:AbstractArray{T}}
    od::D
    method::M
    state::S
    map::MAP_
    nuisance::A
    nlmax::Base.RefValue{T}
    rstar::Base.RefValue{Float64}
    subhess::SubArray{T,2,Array{T,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
end
set!(pl::ProfileLikelihood, v, i) = set!(pl.od.config.f, v, i)
setswap!(map_::MAP, i) = map_.od.config.f.i[] = i
setswap!(pl::ProfileLikelihood, i) = pl.map.od.config.f.i[] = i

struct AsymptoticPosterior{N,T,PL <: ProfileLikelihood{N,T}}
    pl::PL
    state::UnivariateZeroStateBase{T,T,String}
    options::UnivariateZeroOptions{T}
end



MAP(f, ::Val{N}) where N = MAP(f, Vector{Float64}(undef, N+1), Val{N}())

set_identity(x) = Matrix{eltype(x)}(I, length(x), length(x))
function set_identity(invH::AbstractMatrix{T}, x::AbstractArray{T}) where T
    for i in eachindex(x)
        for j in 1:i-1
            invH[j,i] = zero(T)
        end
        invH[i,i] = one(T)
        for j in i+1:length(x)
            invH[j,i] = zero(T)
        end
    end            
end


function MAP(f::F, initial_x::AbstractArray{T}, ::Val{N}) where {F,T,N}

    swap = Swap{N,F}(f, Ref(N))
    # od = OnceDifferentiable(config.swap.f, df, fdf, initial_x, zero(T), Val{true}())
    od = LeanDifferentiable(swap, Val{N}())
    # LBFGS(linesearch=LineSearches.BackTracking())

    backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), set_identity, Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)

    options = LightOptions()
    state = uninitialized_state(nuisance)

    cov_mat = Matrix{T}(undef, N, N)

    map_ = MAP(od, backtrack, options, state, similar(initial_x), initial_x, Ref(lmax), Ref(base_adjust), Vector{T}(undef, N), cov_mat, Val{N}())

    fit!(map_, initial_x)
end
function MAP(od::D, method::M, options::O, state::S, θhat::Vector{T}, buffer::Vector{T}, nlmax::RefValue{T}, base_adjust::RefValue{T}, std_estimates::Vector{T}, cov_mat::Matrix{T}, ::Val{N}) where {D,M,O,S,T,N}
    MAP{N,T,D,M,O,S}(od, method, options, state, θhat, buffer, nlmax, base_adjust, std_estimates, cov_mat)
end



gradient(pl::ProfileLikelihood) = DiffResults.gradient(pl.map.od.config.result)
hessian(pl::ProfileLikelihood) = DiffResults.hessian(pl.map.od.config.result)
gradient(map_::MAP) = DiffResults.gradient(map_.od.config.result)
hessian(map_::MAP) = DiffResults.hessian(map_.od.config.result)

fit!(map_::MAP) = fit!(map_, map_.buffer)
function fit!(map_::MAP{N}, initial_x) where N
    setswap!(map_, N)
    initial_state!(map_.state, map_.method, map_.od, initial_x)
    θhat, nlmax = optimize_light(map_.od, initial_x, map_.method, map_.options, map_.state)
    map_.nlmax[] = nlmax
    copyto!(map_.θhat, θhat)

    hessian!(map.od.config, θhat) #Should update map_.cov_mat

    map_.base_adjust[] = 1 / choldet!(map_.cov, hessian(map_), UpperTriangular, Val{N}())
    Compat.LinearAlgebra.LAPACK.potri!('U', map_.cov) # potri! = chol2inv!
    @inbounds for i ∈ 1:N #Fill in bottom half, because gemm/gemv are much faster than symm/symv?
        for j ∈ 1:i-1
            map_.cov[i,j] = map_.cov[j,i]
        end
        map_.std_estimates[i] = sqrt(map_.cov[i,i])
    end
    map_
end

function set_profile_cov(x::AbstractArray{T}) where T
    Matrix{T}(undef, length(x), length(x))
end
set_profile_cov(x, y) = nothing #Skip in the profile update, so we can manually set it to 


function ProfileLikelihood(f::F, map_::MAP, initial_x::A, ::Val{N}) where {F,T,A<:AbstractArray{T},N}

    profile = ProfileWrapper{N,F,T,A}(f, similar(initial_x), Ref(N), Ref(zero(T)))
    od = LeanDifferentiable(profile, ValM1(Val{N}()))
    # LBFGS(linesearch=LineSearches.BackTracking())

    backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), set_profile_cov, Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)

    nuisance = Vector{T}(N-1)

    # options = LightOptions()
    state = uninitialized_state(nuisance)

    ProfileLikelihood(od, backtrack, state, map_, nuisance, Val{N}())

    # fit!(map_, initial_x)
end
function ProfileLikelihood(od::D, method::M, state::S, map_::MAP, nuisance::A, ::Val{N}) where {N,D,M,S,T,MAP_ <: MAP{N,T},A<:AbstractArray{T}}
    ProfileLikelihood{N,T,D,M,S,MAP_,A}(od, method, state, map_, nuisance, Ref{T}(), Ref{Float64}(), @view(hessian(map_)[1:N-1,1:N-1]))
end

function AsymptoticPosterior()

    options = UnivariateZeroOptions(T,zero(T),1e-7,4eps(T),1e-6)
    state = UnivariateZeroStateBase(zero(T),zero(T),zero(T),zero(T),0,2,false,false,false,false,"")
    AsymptoticPosterior{}(pl, Ref{Float64}())
end

profile_ind(pl::ProfileLikelihood) = pl.od.config.f.i[]
profile_val(pl::ProfileLikelihood) = pl.od.config.f.v[]

function expected_nuisance!(pl::ProfileLikelihood{N}, x, i::Int = profile_ind(pl)) where N
    sf = (x - pl.map.θhat[i]) / pl.map.cov[i,i]
    for j in 1:i-1
        pl.nuisance[j] = pl.map.θhat[j] + pl.map.cov[j,i] * sf
    end
    for j in i+1:N
        pl.nuisance[j-1] = pl.map.θhat[j] + pl.map.cov[j,i] * sf
    end
    nothing
end

function setinvH(invH::AbstractMatrix{T}, cov::AbstractMatrix{T}, i::Int) where T
    n = size(cov,1)
    @inbounds begin
        for j in 1:i-1
            for k in 1:i-1
                invH[k,j] = cov[k,j]
            end
            for k in i+1:n
                invH[k-1,j] = cov[k,j]
            end
        end
        for j in i+1:n
            for k in 1:i-1
                invH[k,j-1] = cov[k,j]
            end
            for k in i+1:n
                invH[k-1,j-1] = cov[k,j]
            end
        end
    end
end

function pdf(pl::ProfileLikelihood, x)
    i = profile_ind(pl::ProfileLikelihood)
    expected_nuisance!(pl, x, i)
    initial_state!(pl.state, pl.method, pl.od, pl.nuisance)
    setinvH!(pl.state.invH, pl.map.cov, i) # Approximate inverse hessian with hessian at maximum.
    θhat, nlmax = optimize_light(pl.od, pl.nuisance, pl.method, pl.map.options, pl.state)
    copyto!(pl.nuisance, θhat)
    pl.nlmax[] = nlmax
end
function pdf(pl::ProfileLikelihood, x, i)
    set!(pl, x, i)
    expected_nuisance!(pl, x, i)
    initial_state!(pl.state, pl.method, pl.od, pl.nuisance)
    setinvH!(pl.state.invH, pl.map.cov, i) # Approximate inverse hessian with hessian at maximum.
    θhat, nlmax = optimize_light(pl.od, pl.nuisance, pl.method, pl.map.options, pl.state)
    copyto!(pl.nuisance, θhat)
    pl.nlmax[] = nlmax
end

function (pl::ProfileLikelihood)(theta, i = profile_ind(pl))
    rstar_p(pl, theta, i) - pl.rstar[]
end

function rp(pl::ProfileLikelihood, x, i = profile_ind(pl))
    sign(pl.map.θhat[i] - x)*sqrt(2(pl.nlmax[]-pl.map.nlmax[]))
end

function rstar_p(pl::ProfileLikelihood, theta, i = profile_ind(pl))
    pdf(pl, theta, i)
    set_buffer_to_profile!(pl, i)
    setswap!(pl, i)
    hessian!(pl.map.od.config, pl.map.buffer)

    r = rp(pl, theta, i)
    r + log(qb(pl, theta, i)/r)/r
end


function qb(pl::ProfileLikelihood{N}, theta, i = profile_ind(pl)) where N
    grad = gradient(pl)
    rootdet = choldet(pl.subhess, UpperTriangular, ValM1(Val{N}()))
    LinearAlgebra.LAPACK.potri!('U', pl.subhess)

    prof_factor = zero(T)
    @inbounds for i in 1:N-1
        for j in 1:i-1
            prof_factor += pl.subhess[j,i] * (grad[j]*hessian(pl)[i,end] + grad[i]*hessian(pl)[j,end])
        end
        prof_factor += pl.subhess[i,i] * grad[i] * hessian(pl)[i,end]
    end
    (prof_factor + grad[N]) * rootdet * pl.map.base_adjust[]
end

function set_buffer_to_profile!(pl::ProfileLikelihood{N}, i = profile_ind(pl)) where N
    @inbounds for j in 1:i-1
        map.buffer[j] = pl.nuisance[j]
    end
    map.buffer[i] = profile_val(pl)
    for j in i+1:N
        map.buffer[j] = pl.nuisance[j-1]
    end
end

Φ⁻¹(x) = √2*erfinv(2x-1)
function quantile(ap::AsymptoticPosterior, alpha, i = profile_ind(ap.pl))
    ap.pl.rstar[] = Φ⁻¹(alpha)
    quantile(ap, i)
end
function quantile(ap::AsymptoticPosterior, i::Int = profile_ind(pl))
    x1 = ap.pl.map.θhat[i] + ap.pl.rstar[] * ap.pl.map.std_estimates[i]
    fx1 = ap.pl(x1)
    step = sign(ap.pl.rstar[]) * fx1 / ap.pl.map.std_estimates[i] 
    x2 = x1 + step

    x2 = 
    find_zero!(ap.state, ap.pl, x0, FalsePosition(), ap.options)
end