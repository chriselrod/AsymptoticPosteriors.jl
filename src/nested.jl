

"""Here, we take a much more straightforward approach to the implementation of Reid."""
struct MAP{N,T,F,D<:AutoDiffDifferentiable{N,T,F},M,O,S}
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

struct ProfileLikelihood{N,T,F,D<:AutoDiffDifferentiable{N,T,<:ProfileWrapper{N,F}},M,S,MAP_ <: MAP{N,T},A<:AbstractArray{T}}
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

struct AsymptoticPosterior{N,T,F,PL <: ProfileLikelihood{N,T,F}}
    pl::PL
    state::UnivariateZeroStateBase{T,T,String}
    options::UnivariateZeroOptions{T}
end

function AsymptoticPosterior(pl::PL,state::UnivariateZeroStateBase{T,T,String},options::UnivariateZeroOptions{T}) where {N,T,F,PL <: ProfileLikelihood{N,T,F}}
    AsymptoticPosterior{N,T,F,PL}(pl,state,options)
end

MAP(f, ::Val{N}) where N = MAP(f, Vector{Float64}(undef, N+1), Val{N}())

set_identity(x) = Matrix{eltype(x)}(I, length(x), length(x))
function set_identity(invH::AbstractMatrix{T}, x::AbstractArray{T}) where T
    @inbounds for i in eachindex(x)
        for j in 1:i-1
            invH[j,i] = zero(T)
        end
        invH[i,i] = one(T)/10^2
        for j in i+1:length(x)
            invH[j,i] = zero(T)
        end
    end            
end

function MAP(f, initial_x::AbstractArray{T}, ::Val{N}) where {T,N}

    # od = OnceDifferentiable(config.swap.f, df, fdf, initial_x, zero(T), Val{true}())
    od = LeanDifferentiable(Swap(f, Val{N}()), Val{N}())
    # Optim.LBFGS(linesearch=LineSearches.BackTracking())

    # backtrack = Optim.BFGS(LineSearches.InitialStatic(), LineSearches.HagerZhang(), set_identity, Optim.Flat())
    backtrack = Optim.BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), set_identity, Optim.Flat())
    # backtrack = Optim.LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)

    options = LightOptions()
    state = uninitialized_state(initial_x)

    cov_mat = Matrix{T}(undef, N, N)

    map_ = MAP(od, backtrack, options, state, similar(initial_x), similar(initial_x), Ref{T}(), Ref{T}(), Vector{T}(undef, N), cov_mat, Val{N}())

    fit!(map_, initial_x)
end
function MAP(od::D, method::M, options::O, state::S, θhat::Vector{T}, buffer::Vector{T}, nlmax::Base.RefValue{T}, base_adjust::Base.RefValue{T}, std_estimates::Vector{T}, cov_mat::Matrix{T}, ::Val{N}) where {N,T,F,D<:AutoDiffDifferentiable{N,T,F},M,O,S}
    MAP{N,T,F,D,M,O,S}(od, method, options, state, θhat, buffer, nlmax, base_adjust, std_estimates, cov_mat)
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
    # @show θhat
    # @show nlmax
    map_.nlmax[] = nlmax
    copyto!(map_.θhat, θhat)

    hessian!(map_.od.config, θhat) #Should update map_.cov_mat

    # @show inv(hessian(map_))
    # @show hessian(map_)
    map_.base_adjust[] = 1 / choldet!(map_.cov, hessian(map_), UpperTriangular, Val{N}())
    Compat.LinearAlgebra.LAPACK.potri!('U', map_.cov) # potri! = chol2inv!
    @inbounds for i ∈ 1:N #Fill in bottom half, because gemm/gemv are much faster than symm/symv?
        for j ∈ 1:i-1
            map_.cov[i,j] = map_.cov[j,i]
        end
        map_.std_estimates[i] = sqrt(map_.cov[i,i])
    end

    # @show abs2.(map_.std_estimates)
    map_
end

function set_profile_cov(x::AbstractArray{T}) where T
    Matrix{T}(undef, length(x), length(x))
end
set_profile_cov(x, y) = nothing #Skip in the profile update, so we can manually set it to 

function ProfileDifferentiable(f::F, x::A, ::Val{N}) where {F,T,A<:AbstractArray{T},N}

    result = DiffResults.HessianResult(x)
    chunk = Chunk(ValM1(Val{N}()))
    gconfig = ForwardDiff.GradientConfig(nothing, x, chunk, ForwardDiff.Tag(nothing, T))
    profile = ProfileWrapper{N,F,T,A,Vector{eltype(gconfig)}}(f, Vector{T}(undef,N), Vector{eltype(gconfig)}(undef,N), Ref(N), Ref(zero(T)))

    # tag = ForwardDiff.Tag(f, T)
    # jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
    # gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    # inner_result = DiffResults.GradientResult(jacobian_config.duals[2])

    gradconfig = GradientConfiguration(profile, result, gconfig)

    GDifferentiable(similar(x), similar(x), gradconfig, Val{N}())
end

function ProfileLikelihood(f::F, map_::MAP, initial_x::A, ::Val{N}, LS = LineSearches.BackTracking()) where {F,T,A<:AbstractArray{T},N}

    nuisance = Vector{T}(undef, N-1)
    od = ProfileDifferentiable(f, nuisance, Val{N}())
    # Optim.LBFGS(linesearch=LineSearches.BackTracking())

    # backtrack = Optim.BFGS(LineSearches.InitialStatic(), LineSearches.HagerZhang(), set_profile_cov, Optim.Flat())
    backtrack = Optim.BFGS(LineSearches.InitialStatic(), LS, set_profile_cov, Optim.Flat())
    # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)


    # options = LightOptions()
    state = uninitialized_state(nuisance)

    ProfileLikelihood(od, backtrack, state, map_, nuisance, Val{N}())

    # fit!(map_, initial_x)
end
function ProfileLikelihood(od::D, method::M, state::S, map_::MAP_, nuisance::A, ::Val{N}) where {N,T,F,D<:AutoDiffDifferentiable{N,T,<:ProfileWrapper{N,F}},M,S,MAP_ <: MAP{N,T},A<:AbstractArray{T}}
    ProfileLikelihood{N,T,F,D,M,S,MAP_,A}(od, method, state, map_, nuisance, Ref{T}(), Ref{Float64}(), @view(hessian(map_)[1:N-1,1:N-1]))
end

#Convenience function does a dynamic dispatch.
AsymptoticPosterior(f, initial_x::AbstractArray) = AsymptoticPosterior(f, initial_x, Val(length(initial_x)))

function AsymptoticPosterior(f::F, initial_x::AbstractArray{T}, ::Val{N}, LS = LineSearches.BackTracking()) where {N,T,F}

    map_ = MAP(f, initial_x, Val{N}())

    pl = ProfileLikelihood(f, map_, initial_x, Val{N}(), LS)

    options = UnivariateZeroOptions(T,zero(T),1e-7,4eps(T),1e-6)
    state = UnivariateZeroStateBase(zero(T),zero(T),zero(T),zero(T),0,2,false,false,false,false,"")
    AsymptoticPosterior(pl, state, options)
end

profile_ind(pl::ProfileLikelihood) = pl.od.config.f.i[]
profile_val(pl::ProfileLikelihood) = pl.od.config.f.v[]


"""
Calculates what the profile expected value would be, given normality.
That is
pl.nuisance = pl.map.θhat_2 + Cov_{2,1}*Cov_{1,1}^{-1}*(x - pl.map.θhat_1)
"""
function normal_expected_nuisance!(pl::ProfileLikelihood{N}, x, i::Int = profile_ind(pl)) where N
    @inbounds begin
        sf = (x - pl.map.θhat[i]) / pl.map.cov[i,i]
        for j in 1:i-1
            pl.nuisance[j] = pl.map.θhat[j] + pl.map.cov[j,i] * sf
        end
        for j in i+1:N
            pl.nuisance[j-1] = pl.map.θhat[j] + pl.map.cov[j,i] * sf
        end
    end
    nothing
end

"""
Naively assumes indepedence, and just uses the global maximum for the profile maximum.
This seems like it ought to perform poorly, but in tests it appears more robust.
"""
function naive_expected_nuisance!(pl::ProfileLikelihood{N}, x, i::Int = profile_ind(pl)) where N
    @inbounds begin
        # sf = (x - pl.map.θhat[i]) / pl.map.cov[i,i]
        for j in 1:i-1
            pl.nuisance[j] = pl.map.θhat[j]# + pl.map.cov[j,i] * sf
        end
        for j in i+1:N
            pl.nuisance[j-1] = pl.map.θhat[j]# + pl.map.cov[j,i] * sf
        end
    end
    nothing
end

function setinvH!(invH::AbstractMatrix{T}, cov::AbstractMatrix{T}, i::Int) where T
    n = size(cov,1)
    divisor = 10
    @inbounds begin
        for j in 1:i-1
            for k in 1:i-1
                invH[k,j] = cov[k,j]/divisor
            end
            for k in i+1:n
                invH[k-1,j] = cov[k,j]/divisor
            end
        end
        for j in i+1:n
            for k in 1:i-1
                invH[k,j-1] = cov[k,j]/divisor
            end
            for k in i+1:n
                invH[k-1,j-1] = cov[k,j]/divisor
            end
        end
    end
end

function profilepdf(pl::ProfileLikelihood, x, ::Val{reset_search}=Val{true}()) where reset_search
    i = profile_ind(pl::ProfileLikelihood)
    # expected_nuisance!(pl, x, i)
    reset_search && naive_expected_nuisance!(pl, x, i)
    initial_state!(pl.state, pl.method, pl.od, pl.nuisance)
    setinvH!(pl.state.invH, pl.map.cov, i) # Approximate inverse hessian with hessian at maximum.
    θhat, nlmax = optimize_light(pl.od, pl.nuisance, pl.method, pl.map.options, pl.state)
    copyto!(pl.nuisance, θhat)
    pl.nlmax[] = nlmax
end
function profilepdf(pl::ProfileLikelihood, x, i, ::Val{reset_search}=Val{true}()) where reset_search
    set!(pl, x, i)
    # expected_nuisance!(pl, x, i)
    reset_search && naive_expected_nuisance!(pl, x, i)
    initial_state!(pl.state, pl.method, pl.od, pl.nuisance)
    setinvH!(pl.state.invH, pl.map.cov, i) # Approximate inverse hessian with hessian at maximum.
    θhat, nlmax = optimize_light(pl.od, pl.nuisance, pl.method, pl.map.options, pl.state)
    copyto!(pl.nuisance, θhat)
    pl.nlmax[] = nlmax
end

function (pl::ProfileLikelihood)(theta, i = profile_ind(pl), ::Val{reset_search}=Val{false}()) where reset_search
    debug() && println("")
    debug() && @show theta, pl.rstar[], i
    # @assert isnan(theta) == false
    debug() && @show rstar_p(pl, theta, i) + pl.rstar[]
    rstar_p(pl, theta, i, Val{reset_search}()) + pl.rstar[]
end

# Positive if x is smaller than mode, negative if bigger
function rp(pl::ProfileLikelihood, x, i = profile_ind(pl))
    copysign( sqrt(2(pl.nlmax[]-pl.map.nlmax[])), pl.map.θhat[i] - x)
end

function rstar_p(pl::ProfileLikelihood{N}, theta, i::Int=profile_ind(pl), ::Val{reset_search}=Val{true}()) where {N,reset_search}

    profilepdf(pl, theta, i, Val{reset_search}())
    set_buffer_to_profile!(pl, i)

    # pl.map.buffer[i], pl.map.buffer[N] = pl.map.buffer[N], pl.map.buffer[i]
    # setswap!(pl, N)
    # hessian!(pl.map.od.config, pl.map.buffer)
    # @show hessian(pl)
    # set_buffer_to_profile!(pl, i)

    setswap!(pl, i)
    hessian!(pl.map.od.config, pl.map.buffer)
    # @show hessian(pl)
    debug() && @show pl.map.buffer
    debug() && @show hessian(pl)

    r = rp(pl, theta, i)
    # println("\n\n\nCalling qb:")
    # @show r
    # qb_res = qb(pl, theta, i)
    # @show pl.rstar[]
    # @show r
    # @show qb_res
    # @show r + log(qb_res/r)/r
    r + log(qb(pl, theta, i)/r)/r
end

# @inline function fdf_adjrstar_p(ap::AsymptoticPosterior, theta, i::Int=profile_ind(pl),::Val{reset_search}=Val{true}()) where reset_search
#     fdf_adjrstar_p(ap.pl, theta, i,Val{reset_search}())
# end
function fdf_adjrstar_p(pl::ProfileLikelihood{N,T}, theta, i::Int=profile_ind(pl),::Val{reset_search}=Val{true}()) where {N,T,reset_search}

    profilepdf(pl, theta, i, Val{reset_search}())
    set_buffer_to_profile!(pl, i)
    setswap!(pl, i)
    hessian!(pl.map.od.config, pl.map.buffer)
    
    delta_log_likelihood = pl.nlmax[]-pl.map.nlmax[]
    r = copysign( sqrt(2delta_log_likelihood), pl.map.θhat[i] - theta)
    
    grad = gradient(pl)

    rootdet = choldet!(pl.subhess, UpperTriangular, ValM1(Val{N}()))
    LinearAlgebra.LAPACK.potri!('U', pl.subhess)

    prof_factor = zero(T)
    @inbounds for i in 1:N-1
        hᵢₙ = hessian(pl)[i,end]
        gᵢ = grad[i]
        for j in 1:i-1
            prof_factor += pl.subhess[j,i] * (grad[j]*hᵢₙ + gᵢ*hessian(pl)[j,end])
        end
        prof_factor += pl.subhess[i,i] * gᵢ * hᵢₙ
    end
    hess_adjust = rootdet * pl.map.base_adjust[]
    q = (prof_factor - grad[N]) * hess_adjust

    rstar = r + log(q/r)/r
    rstar + pl.rstar[], exp(abs2(rstar)/2-delta_log_likelihood) / hess_adjust
end
function pdf(pl::ProfileLikelihood{N,T}, theta, i = profile_ind(pl),::Val{reset_search}=Val{true}()) where {N,T,reset_search}

    profilepdf(pl, theta, i, Val{reset_search}())
    set_buffer_to_profile!(pl, i)
    setswap!(pl, i)
    hessian!(pl.map.od.config, pl.map.buffer)    

    rootdet = choldet!(pl.subhess, UpperTriangular, ValM1(Val{N}())) #is there a more efficient way?

    exp(pl.map.nlmax[]-pl.nlmax[]) / (sqrt(2π) * rootdet * pl.map.base_adjust[])
end


function qb(pl::ProfileLikelihood{N,T}, theta, i = profile_ind(pl)) where {N,T}
    grad = gradient(pl)

    # @show pl.map.θhat[i]
    # @show theta

    # @show abs2.(pl.map.std_estimates)
    # @show inv(hessian(pl))
    # @show inv(hessian(pl))
    # @show pl.subhess

    # prof_p_g2 = grad' * vcat( inv(hessian(pl)[1:N-1,1:N-1]) * hessian(pl)[1:N-1,end], -1.0  )
    # prof_2 = grad[1:N-1]' *  inv(hessian(pl)[1:N-1,1:N-1]) * hessian(pl)[1:N-1,end]
    # @show prof_p_g2
    # @show prof_2
    # @show prof_2 - grad[end]
    # ForwardDiff.gradient(f2, prex_max_dens4_)' * vcat(-inv(BigFloat.(precise_hessian2[1:end-1,1:end-1]))*BigFloat.(precise_hessian2[1:end-1,end]),big(1.0))

    rootdet = choldet!(pl.subhess, UpperTriangular, ValM1(Val{N}()))
    LinearAlgebra.LAPACK.potri!('U', pl.subhess)

    prof_factor = zero(T)
    @inbounds for i in 1:N-1
        hᵢₙ = hessian(pl)[i,end]
        gᵢ = grad[i]
        for j in 1:i-1
            prof_factor += pl.subhess[j,i] * (grad[j]*hᵢₙ + gᵢ*hessian(pl)[j,end])
        end
        prof_factor += pl.subhess[i,i] * gᵢ * hᵢₙ
    end
    # @show -prof_factor
    # @show grad[N]
    # @show (prof_factor - grad[N])
    # @show rootdet
    # @show pl.map.base_adjust[]
    (prof_factor - grad[N]) * rootdet * pl.map.base_adjust[]
end

# function set_buffer_to_profile!(pl::ProfileLikelihood{N}, i = profile_ind(pl)) where N
#     @inbounds begin
#         for j in 1:i-1
#             pl.map.buffer[j] = pl.nuisance[j]
#         end
#         pl.map.buffer[i] = profile_val(pl)
#         for j in i+1:N
#             pl.map.buffer[j] = pl.nuisance[j-1]
#         end
#     end
# end
function set_buffer_to_profile!(pl::ProfileLikelihood{N}, i = profile_ind(pl)) where N
    @inbounds begin # turn these off while testing...or start with inbounds=no
        for j in 1:i-1
            pl.map.buffer[j] = pl.nuisance[j]
        end
        if i != N
            pl.map.buffer[i] = pl.nuisance[N-1]
            for j in i+1:N-1
                pl.map.buffer[j] = pl.nuisance[j-1]
            end
        end
        pl.map.buffer[N] = profile_val(pl)
    end
end

Φ⁻¹(x) = √2*erfinv(2x-1)
function Base.quantile(ap::AsymptoticPosterior, alpha, i)
    ap.pl.od.config.f.i[] = i
    ap.pl.rstar[] = Φ⁻¹(alpha)
    # quadratic_search(ap, i)
    linear_search(ap, profile_ind(ap.pl))
end
function Base.quantile(ap::AsymptoticPosterior, alpha)
    ap.pl.rstar[] = Φ⁻¹(alpha)
    linear_search(ap, profile_ind(ap.pl))
    # quadratic_search(ap, profile_ind(ap.pl))
end


export lquantile, qquantile
function lquantile(ap::AsymptoticPosterior, alpha, i)
    ap.pl.od.config.f.i[] = i
    ap.pl.rstar[] = Φ⁻¹(alpha)
    linear_search(ap, i)
end
function lquantile(ap::AsymptoticPosterior, alpha)
    ap.pl.rstar[] = Φ⁻¹(alpha)
    linear_search(ap, profile_ind(ap.pl))
end

function qquantile(ap::AsymptoticPosterior, alpha, i)
    ap.pl.od.config.f.i[] = i
    ap.pl.rstar[] = Φ⁻¹(alpha)
    quadratic_search(ap, i)
end
function qquantile(ap::AsymptoticPosterior, alpha)
    ap.pl.rstar[] = Φ⁻¹(alpha)
    quadratic_search(ap, profile_ind(ap.pl))
end
