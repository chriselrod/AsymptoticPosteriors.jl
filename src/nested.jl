

"""Here, we take a much more straightforward approach to the implementation of Reid."""
struct MAP{P,T,D<:DifferentiableObject{P},M,O,S,L}
    od::D
    method::M
    options::O
    state::S
    θhat::SizedSIMDVector{P,T}
    buffer::SizedSIMDVector{P,T}
    nlmax::Base.RefValue{T}
    base_adjust::Base.RefValue{T}
    std_estimates::SizedSIMDVector{P,T}
    cov::SymmetricMatrix{T,P,L}
end

struct ProfileLikelihood{P,T,Pm1,D<:DifferentiableObject{Pm1},M,S,MAP_ <: MAP{P,T}}
    od::D
    method::M
    state::S
    map::MAP_
    nuisance::SizedSIMDVector{Pm1,T}#A
    nlmax::Base.RefValue{T}
    rstar::Base.RefValue{Float64}
    # subhess::SubArray{T,2,Array{T,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
end
set!(pl::ProfileLikelihood, v, i) = set!(pl.od.config.f, v, i)
setswap!(map_::MAP, i) = map_.od.config.f.i[] = i
setswap!(pl::ProfileLikelihood, i) = pl.map.od.config.f.i[] = i

struct AsymptoticPosterior{P,T,PL <: ProfileLikelihood{P,T}}
    pl::PL
    state::UnivariateZeroStateBase{T,T}
    options::UnivariateZeroOptions{T}
end

function AsymptoticPosterior(pl::PL,state::UnivariateZeroStateBase{T,T},options::UnivariateZeroOptions{T}) where {P,T,PL <: ProfileLikelihood{P,T}}
    AsymptoticPosterior{P,T,PL}(pl,state,options)
end

# @generated function MAP(f, ::Val{P}) where P
#     :(MAP(f, SizedSIMDVector{$(P+1),Float64}(undef)))
# end

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

function MAP(f, initial_x::SizedSIMDVector{P,T}) where {T,P}

    # od = OnceDifferentiable(config.swap.f, df, fdf, initial_x, zero(T), Val{true}())
    od = TwiceDifferentiable(Swap(f, Val{P}()), Val{P}())
    # Optim.LBFGS(linesearch=LineSearches.BackTracking())

    # backtrack = Optim.BFGS(LineSearches.InitialStatic(), LineSearches.HagerZhang(), set_identity, Optim.Flat())
    backtrack = BFGS( linesearch = LineSearches.BackTracking())
    # backtrack = Optim.LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)

    options = LightOptions()
    state = DifferentiableObjects.uninitialized_state(initial_x)

    cov_mat = SymmetricMatrix{T,P}(undef)

    map_ = MAP(od, backtrack, options, state, similar(initial_x), similar(initial_x), Ref{T}(), Ref{T}(), similar(initial_x), cov_mat)
end
function MAP(od::D, method::M, options::O, state::S, θhat::SizedSIMDVector{P,T}, buffer::SizedSIMDVector{P,T}, nlmax::Base.RefValue{T}, base_adjust::Base.RefValue{T}, std_estimates::SizedSIMDVector{P,T}, cov_mat::SymmetricMatrix{T,P,L}) where {P,T,D<:DifferentiableObject{P},M,O,S,L}
    MAP{P,T,D,M,O,S,L}(od, method, options, state, θhat, buffer, nlmax, base_adjust, std_estimates, cov_mat)
end



gradient(pl::ProfileLikelihood) = DiffResults.gradient(pl.map.od.config.result)
hessian(pl::ProfileLikelihood) = DiffResults.hessian(pl.map.od.config.result)
gradient(map_::MAP) = DiffResults.gradient(map_.od.config.result)
hessian(map_::MAP) = DiffResults.hessian(map_.od.config.result)

fit!(map_::MAP) = fit!(map_, map_.buffer)
function fit!(map_::MAP{P}, initial_x) where P
    setswap!(map_, P)
    DifferentiableObjects.initial_state!(map_.state, map_.method, map_.od, initial_x)
    θhat, nlmax = optimize_light(map_.od, initial_x, map_.method, map_.options, map_.state)
    # @show θhat
    # @show nlmax
    map_.nlmax[] = nlmax
    copyto!(map_.θhat, θhat)

    hessian!(map_.od.config, θhat) 

    # @show inv(hessian(map_))
    # @show hessian(map_)
    map_.base_adjust[] = 1 / invdet!(map_.cov, hessian(map_))
    # inv!(map_.cov) # potri! = chol2inv!
    # xxt!(map_.cov)
    @inbounds for i ∈ 1:P
        #Fill in bottom half, because gemm/gemv are much faster than symm/symv?
        # for j ∈ 1:i-1
        #     map_.cov[i,j] = map_.cov[j,i]
        # end
        map_.std_estimates[i] = sqrt(map_.cov[i,i])
    end

    # @show abs2.(map_.std_estimates)
    map_
end

# set_profile_cov(x::SizedSIMDVector{P,T}) where {T,P} = SymmetricMatrix{T,P}()
# #Skip in the profile update, so we can manually set it to 
# set_profile_cov(x, y) = nothing 

function ProfileDifferentiable(f::F, x::SizedSIMDVector{Pm1,T}, ::Val{P}) where {F,T,Pm1,P}

    result = DiffResults.GradientResult(x)
    # result = DiffResults.HessianResult(x)
    chunk = DifferentiableObjects.Chunk(Val{Pm1}())
    gconfig = ForwardDiff.GradientConfig(nothing, x, chunk, ForwardDiff.Tag(nothing, T))
    profile = ProfileWrapper{P,F,T,eltype(gconfig)}(f,
        SizedSIMDVector{P,T}(undef),
        SizedSIMDVector{P,eltype(gconfig)}(),
        Ref(P), Ref(zero(T))
    )

    # tag = ForwardDiff.Tag(f, T)
    # jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
    # gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    # inner_result = DiffResults.GradientResult(jacobian_config.duals[2])

    gradconfig = GradientConfiguration(profile, result, gconfig)

    OnceDifferentiable(similar(x), similar(x), gradconfig)#, Val{P}())
end

@generated function ProfileLikelihood(f, map_::MAP, initial_x::SizedSIMDVector{P,T}, LS = LineSearches.BackTracking()) where {T,P}
    quote
        nuisance = SizedSIMDVector{$(P-1),$T}()
        od = ProfileDifferentiable(f, nuisance, Val{$P}())
        # Optim.LBFGS(linesearch=LineSearches.BackTracking())

        # backtrack = Optim.BFGS(LineSearches.InitialStatic(), LineSearches.HagerZhang(), set_profile_cov, Optim.Flat())
        backtrack = BFGS( linesearch = LS )#, set_profile_cov, Optim.Flat())
        # backtrack = LBFGS(10, LineSearches.InitialStatic(), LineSearches.BackTracking(), nothing, (P, x) -> nothing, Optim.Flat(), true)


        # options = LightOptions()
        state = DifferentiableObjects.uninitialized_state(nuisance)

        ProfileLikelihood(od, backtrack, state, map_, nuisance)
    end
end
function ProfileLikelihood(od::D, method::M, state::S, map_::MAP_, nuisance::SizedSIMDVector{Pm1,T}) where {P,T,Pm1,D<:DifferentiableObject{Pm1},M,S,MAP_ <: MAP{P}}
    ProfileLikelihood{P,T,Pm1,D,M,S,MAP_}(od, method, state, map_, nuisance, Ref{T}(), Ref{Float64}())#, @view(hessian(map_)[1:P-1,1:P-1]))
end

#Convenience function does a dynamic dispatch.
function AsymptoticPosterior(f, initial_x::AbstractArray{T}) where T
    AsymptoticPosterior(f, SizedSIMDVector{length(initial_x),T}(initial_x))
end

# function AsymptoticPosterior(f::F, initial_x::AbstractArray{T}, ::Val{P}, LS = LineSearches.BackTracking()) where {P,T,F}
function AsymptoticPosterior(f, initial_x::SizedSIMDVector{P,T}, LS = LineSearches.HagerZhang()) where {P,T}
    map_ = MAP(f, initial_x)
    fit!(map_, initial_x)
    pl = ProfileLikelihood(f, map_, initial_x, LS)

    options = UnivariateZeroOptions(T,zero(T),1e-7,4eps(T),1e-6)
    state = UnivariateZeroStateBase(zero(T),zero(T),zero(T),zero(T),0,2,false,false,false,false,"")
    AsymptoticPosterior(pl, state, options)
end
function AsymptoticPosterior(::UndefInitializer, f, initial_x::SizedSIMDVector{P,T}, LS = LineSearches.HagerZhang()) where {P,T}
    map_ = MAP(f, initial_x)
    # fit!(map_, initial_x)
    pl = ProfileLikelihood(f, map_, initial_x, LS)

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
function normal_expected_nuisance!(pl::ProfileLikelihood{P}, x, i::Int = profile_ind(pl)) where P
    @inbounds begin
        sf = (x - pl.map.θhat[i]) / pl.map.cov[i,i]
        for j in 1:i-1
            pl.nuisance[j] = pl.map.θhat[j] + pl.map.cov[j,i] * sf
        end
        for j in i+1:P
            pl.nuisance[j-1] = pl.map.θhat[j] + pl.map.cov[j,i] * sf
        end
    end
    nothing
end

"""
Naively assumes indepedence, and just uses the global maximum for the profile maximum.
This seems like it ought to perform poorly, but in tests it appears more robust.
"""
function naive_expected_nuisance!(pl::ProfileLikelihood{P}, x, i::Int = profile_ind(pl)) where P
    @inbounds begin
        # sf = (x - pl.map.θhat[i]) / pl.map.cov[i,i]
        for j in 1:i-1
            pl.nuisance[j] = pl.map.θhat[j]# + pl.map.cov[j,i] * sf
        end
        for j in i+1:P
            pl.nuisance[j-1] = pl.map.θhat[j]# + pl.map.cov[j,i] * sf
        end
    end
    nothing
end

function setinvH!(invH::SymmetricMatrix{T,P}, cov::SymmetricMatrix{T,P}, i::Int) where {T,P}
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
    DifferentiableObjects.initial_state!(pl.state, pl.method, pl.od, pl.nuisance)
    # I'm seeing how just using I as the initial H works out.
    # setinvH!(pl.state.invH, pl.map.cov, i) # Approximate inverse hessian with hessian at maximum.
    θhat, nlmax = optimize_light(pl.od, pl.nuisance, pl.method, pl.map.options, pl.state)
    copyto!(pl.nuisance, θhat)
    pl.nlmax[] = nlmax
end
function profilepdf(pl::ProfileLikelihood, x, i, ::Val{reset_search}=Val{true}()) where reset_search
    set!(pl, x, i)
    # expected_nuisance!(pl, x, i)
    reset_search && naive_expected_nuisance!(pl, x, i)
    DifferentiableObjects.initial_state!(pl.state, pl.method, pl.od, pl.nuisance)
    # I'm seeing how just using I as the initial H works out.
    # setinvH!(pl.state.invH, pl.map.cov, i) # Approximate inverse hessian with hessian at maximum.
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

function rstar_p(pl::ProfileLikelihood{P}, theta, i::Int=profile_ind(pl), ::Val{reset_search}=Val{true}()) where {P,reset_search}

    profilepdf(pl, theta, i, Val{reset_search}())
    set_buffer_to_profile!(pl, i)

    # pl.map.buffer[i], pl.map.buffer[P] = pl.map.buffer[P], pl.map.buffer[i]
    # setswap!(pl, P)
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
@generated function fdf_adjrstar_p(pl::ProfileLikelihood{P,T}, theta, p_i::Int=profile_ind(pl),::Val{reset_search}=Val{true}()) where {P,T,reset_search}

    q, qa = TriangularMatrices.create_quote()
    ind = 0
    for i ∈ 1:P, j ∈ 1:i
        ind += 1
        push!(qa, :( $(Symbol(:hess_, ind)) = hess[$( P*(i-1) + j )]  ))
    end
    TriangularMatrices.extract_linear!(qa, P, :grad)
    stn = TriangularMatrices.small_triangle(P)
    @inbounds for i in 1:P-1
        hᵢₙ = Symbol(:hess_, stn + i) # hessian(pl)[i,end]
        gᵢ = Symbol(:grad_, i)
        sti = TriangularMatrices.small_triangle(i)
        for j in 1:i-1
            push!(qa, :(prof_factor += $(Symbol(:hess_, sti + j)) *
                        ( $(Symbol(:grad_, j)) * $hᵢₙ + $gᵢ*$(Symbol(:hess_, stn + j))  ) ) )
            # prof_factor += pl.subhess[j,i] * (grad[j]*hᵢₙ + gᵢ*hessian(pl)[j,end])
        end
        push!(qa, :(prof_factor += $(Symbol(:hess_, sti + i)) * $gᵢ * $hᵢₙ ))
        # prof_factor += pl.subhess[i,i] * gᵢ * hᵢₙ
    end

    quote
        profilepdf(pl, theta, p_i, Val{reset_search}())
        set_buffer_to_profile!(pl, p_i)
        setswap!(pl, p_i)
        hess = hessian!(pl.map.od.config, pl.map.buffer)
        
        delta_log_likelihood = pl.nlmax[]-pl.map.nlmax[]
        r = copysign( sqrt(2delta_log_likelihood), pl.map.θhat[p_i] - theta)
        
        grad = gradient(pl)

        rootdet = invdet!(hessian(pl), Val{$(P-1)}())
        
        prof_factor = zero(T)
        $q
        # @inbounds for i in 1:P-1
        #     hᵢₙ = hessian(pl)[i,end]
        #     gᵢ = grad[i]
        #     for j in 1:i-1
        #         prof_factor += pl.subhess[j,i] * (grad[j]*hᵢₙ + gᵢ*hessian(pl)[j,end])
        #     end
        #     prof_factor += pl.subhess[i,i] * gᵢ * hᵢₙ
        # end
        hess_adjust = rootdet * pl.map.base_adjust[]
        q = (prof_factor - grad[P]) * hess_adjust

        rstar = r + log(q/r)/r
        rstar + pl.rstar[], exp(abs2(rstar)/2-delta_log_likelihood) / hess_adjust
    end
end
function pdf(pl::ProfileLikelihood{P,T,Pm1}, theta, i = profile_ind(pl),::Val{reset_search}=Val{true}()) where {P,T,Pm1,reset_search}

    profilepdf(pl, theta, i, Val{reset_search}())
    set_buffer_to_profile!(pl, i)
    setswap!(pl, i)
    hessian!(pl.map.od.config, pl.map.buffer)    

    # is there a more efficient way of calculating the determinant?
    # skip storing? Implement that...
    rootdet = choldet!(hessian(pl), Val{Pm1}()) 

    exp(pl.map.nlmax[]-pl.nlmax[]) / (sqrt(2π) * rootdet * pl.map.base_adjust[])
end


@generated function qb(pl::ProfileLikelihood{P,T}, theta, pi = profile_ind(pl)) where {P,T}

    # @show pl.map.θhat[i]
    # @show theta

    # @show abs2.(pl.map.std_estimates)
    # @show inv(hessian(pl))
    # @show inv(hessian(pl))
    # @show pl.subhess

    # prof_p_g2 = grad' * vcat( inv(hessian(pl)[1:P-1,1:P-1]) * hessian(pl)[1:P-1,end], -1.0  )
    # prof_2 = grad[1:P-1]' *  inv(hessian(pl)[1:P-1,1:P-1]) * hessian(pl)[1:P-1,end]
    # @show prof_p_g2
    # @show prof_2
    # @show prof_2 - grad[end]
    # ForwardDiff.gradient(f2, prex_max_dens4_)' * vcat(-inv(BigFloat.(precise_hessian2[1:end-1,1:end-1]))*BigFloat.(precise_hessian2[1:end-1,end]),big(1.0))
    q, qa = TriangularMatrices.create_quote()
    ind = 0
    for i ∈ 1:P-1, j ∈ 1:i
        ind += 1
        push!(qa, :( $(Symbol(:hess_,ind)) = hess[ $( P*(i-1)+j ) ] ) )
    end
    # TriangularMatrices.extract_linear!(qa, P, :hess, :hess, TriangularMatrices.small_triangle(P))
    TriangularMatrices.extract_linear!(qa, P, :grad)
    for i ∈ 1:P-1
        hᵢₙ = Symbol(:hess_, i)
        gᵢ = Symbol(:grad_, i)
        for j ∈ 1:i-1
            push!(qa, :( prof_factor += hess[$(j+TriangularMatrices.small_triangle(i))] * ($(Symbol(:grad_, j))*$hᵢₙ + $gᵢ*$(Symbol(:hess_, j)) ) ) )
        end
        push!(qa, :(prof_factor += hess[$(TriangularMatrices.big_triangle(i))] * $gᵢ * $hᵢₙ))
    end

    quote
        grad = gradient(pl)
        hess = hessian(pl) # P x P
        rootdet = invdet!(hess, Val{$(P-1)}()) # only leeding P-1 x P-1
        prof_factor = zero(T)
        $q
        # @inbounds for i in 1:P-1
        #     hᵢₙ = hessian(pl)[i+$(TriangularMatrices.small_triangle(P))]
        #     gᵢ = grad[i]
        #     for j in 1:i-1
        #         prof_factor += pl.subhess[j,i] * (grad[j]*hᵢₙ + gᵢ*hessian(pl)[j,end])
        #     end
        #     prof_factor += pl.subhess[i,i] * gᵢ * hᵢₙ
        # end
        # @show -prof_factor
        # @show grad[P]
        # @show (prof_factor - grad[P])
        # @show rootdet
        # @show pl.map.base_adjust[]
        (prof_factor - $(Symbol(:grad_, P))) * rootdet * pl.map.base_adjust[]
    end
end

# function set_buffer_to_profile!(pl::ProfileLikelihood{P}, i = profile_ind(pl)) where P
#     @inbounds begin
#         for j in 1:i-1
#             pl.map.buffer[j] = pl.nuisance[j]
#         end
#         pl.map.buffer[i] = profile_val(pl)
#         for j in i+1:P
#             pl.map.buffer[j] = pl.nuisance[j-1]
#         end
#     end
# end
function set_buffer_to_profile!(pl::ProfileLikelihood{P}, i = profile_ind(pl)) where P
    @inbounds begin # start with bounds-check=yes while testing
        for j in 1:i-1
            pl.map.buffer[j] = pl.nuisance[j]
        end
        if i != P
            pl.map.buffer[i] = pl.nuisance[P-1]
            for j in i+1:P-1
                pl.map.buffer[j] = pl.nuisance[j-1]
            end
        end
        pl.map.buffer[P] = profile_val(pl)
    end
end

Φ⁻¹(x) = √2*erfinv(2x-1)
function Statistics.quantile(ap::AsymptoticPosterior, alpha, i)
    ap.pl.od.config.f.i[] = i
    ap.pl.rstar[] = Φ⁻¹(alpha)
    # quadratic_search(ap, i)
    linear_search(ap, profile_ind(ap.pl))
end
function Statistics.quantile(ap::AsymptoticPosterior, alpha)
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


(ap::AsymptoticPosterior)(x) = ap.pl.map.od(x)
mode(ap::AsymptoticPosterior) = ap.pl.map.θhat