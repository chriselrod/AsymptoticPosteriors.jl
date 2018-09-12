

"""Here, we take a much more straightforward approach to the implementation of Reid."""
struct MAP{P,T,R,D<:DifferentiableObject{P},M,S,L}
    od::D
    method::M
    state::S
    θhat::SizedSIMDVector{P,T,R}
    buffer::SizedSIMDVector{P,T,R}
    nlmax::Base.RefValue{T}
    base_adjust::Base.RefValue{T}
    std_estimates::SizedSIMDVector{P,T,R}
    Lfull::SizedSIMDMatrix{P,P,T,R,L}
end

struct AsymptoticPosterior{P,T,Pm1,R,D<:DifferentiableObject{Pm1},M,S,MAP_ <: MAP{P,T,R},SV <: SizedSIMDVector{Pm1,T},SM <: SizedSIMDMatrix{Pm1,Pm1,T}}
    od::D
    method::M
    state::S
    map::MAP_
    nuisance::SV#A
    nlmax::Base.RefValue{T}
    rstar::Base.RefValue{T}
    Lsmall::SM
    # subhess::SubArray{T,2,Array{T,2},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
end
@inline set!(ap::AsymptoticPosterior, v, i) = set!(ap.od.config.f, v, i)
@inline setswap!(map_::MAP, i) = map_.od.config.f.i[] = i
@inline setswap!(ap::AsymptoticPosterior, i) = ap.map.od.config.f.i[] = i

function MAP(f, initial_x::SizedSIMDVector{P,T}) where {T,P}

    od = TwiceDifferentiable(Swap(f, Val{P}()), Val{P}())

    state = DifferentiableObjects.BFGSState2(Val(P))
    backtrack = DifferentiableObjects.BackTracking2(Val(3))

    Lfull = zero(SizedSIMDMatrix{P,P,T})

    map_ = MAP(od, backtrack, state, state.x_old, similar(initial_x), Ref{T}(), Ref{T}(), similar(initial_x), Lfull)
end
function MAP(od::D, method::M, state::S, θhat::SizedSIMDVector{P,T}, buffer::SizedSIMDVector{P,T}, nlmax::Base.RefValue{T}, base_adjust::Base.RefValue{T}, std_estimates::SizedSIMDVector{P,T}, Lfull::SizedSIMDMatrix{P,P,T,R,L}) where {P,T,R,D<:DifferentiableObject{P},M,S,L}
    MAP{P,T,R,D,M,S,L}(od, method, state, θhat, buffer, nlmax, base_adjust, std_estimates, Lfull)
end



@inline gradient(ap::AsymptoticPosterior) = ap.map.od.config.result.grad
@inline hessian(ap::AsymptoticPosterior) = ap.map.od.config.result.hess
@inline gradient(map_::MAP) = map_.od.config.result.grad
@inline hessian(map_::MAP) = map_.od.config.result.hess

fit!(map_::MAP) = fit!(map_, map_.buffer)
function fit!(map_::MAP{P,T}, initial_x) where {P,T}
    setswap!(map_, P)
    # DifferentiableObjects.initial_state!(map_.state, map_.method, map_.od, initial_x)
    scaled_nlmax, scale = optimize_scale!(map_.state, map_.od, initial_x, map_.method, one(T))
    map_.nlmax[] = scaled_nlmax / scale
    θhat = map_.θhat

    hessian!(map_.od.config, θhat)

    # @show inv(hessian(map_))
    # @show hessian(map_)
    map_.base_adjust[] = 1 / invcholdet!(map_.Lfull, hessian(map_))
    # @fastmath @inbounds for i ∈ 1:P
    @inbounds for i ∈ 1:P
        varᵢ = map_.Lfull[i,i] * map_.Lfull[i,i]
        @fastmath for j ∈ i+1:P
            varᵢ += map_.Lfull[j,i] * map_.Lfull[j,i]
        end
        map_.std_estimates[i] = sqrt( varᵢ )
    end

    # @show abs2.(map_.std_estimates)
    map_
end

function ProfileDifferentiable(f::F, x::SizedSIMDVector{Pm1,T}, ::Val{P}) where {F,T,Pm1,P}

    result = DifferentiableObjects.GradientDiffResult(x)
    chunk = DifferentiableObjects.Chunk(Val{Pm1}())
    gconfig = ForwardDiff.GradientConfig(nothing, x, chunk, ForwardDiff.Tag(nothing, T))
    profile = ProfileWrapper{P,F,T,eltype(gconfig)}(f,
        SizedSIMDVector{P,T}(undef),
        SizedSIMDVector{P,eltype(gconfig)}(undef),
        Ref(P), Ref(zero(T))
    )

    gradconfig = GradientConfiguration(profile, result, gconfig)

    OnceDifferentiable(similar(x), similar(x), gradconfig)#, Val{P}())
end

@generated function AsymptoticPosterior(f, map_::MAP, initial_x::SizedSIMDVector{P,T}) where {T,P}
    quote

        backtrack = DifferentiableObjects.BackTracking2(Val(3))
        # state = DifferentiableObjects.uninitialized_state(nuisance)
        state = DifferentiableObjects.BFGSState2(Val{$(P-1)}())
        od = ProfileDifferentiable(f, state.x_old, Val{$P}())

        AsymptoticPosterior(od, backtrack, state, map_, state.x_old, SizedSIMDMatrix{$(P-1),$(P-1),$T}(undef))
    end
end
function AsymptoticPosterior(od::D, method::M, state::S, map_::MAP_, nuisance::SV, Lsmall::SM) where {P,T,Pm1,R,D<:DifferentiableObject{Pm1},M,S,MAP_ <: MAP{P,T,R},SV <: SizedSIMDVector{Pm1,T},SM <: SizedSIMDMatrix{Pm1,Pm1,T}}

    AsymptoticPosterior{P,T,Pm1,R,D,M,S,MAP_,SV,SM}(od, method, state, map_, nuisance, Ref{T}(), Ref{T}(), Lsmall)#, @view(hessian(map_)[1:P-1,1:P-1]))
end

#Convenience function does a dynamic dispatch.
function AsymptoticPosterior(f, initial_x::AbstractArray{T}) where T
    AsymptoticPosterior(f, SizedSIMDVector{length(initial_x),T}(initial_x))
end

function AsymptoticPosterior(f, initial_x::SizedSIMDVector{P,T}) where {P,T}
    map_ = MAP(f, initial_x)
    fit!(map_, initial_x)
    AsymptoticPosterior(f, map_, initial_x)
end
function AsymptoticPosterior(::UndefInitializer, f, initial_x::SizedSIMDVector{P,T}) where {P,T}
    map_ = MAP(f, initial_x)
    # fit!(map_, initial_x)
    AsymptoticPosterior(f, map_, initial_x)
end

profile_ind(ap::AsymptoticPosterior) = ap.od.config.f.i[]
profile_val(ap::AsymptoticPosterior) = ap.od.config.f.v[]


# """
# Calculates what the profile expected value would be, given normality.
# That is
# ap.nuisance = ap.map.θhat_2 + Cov_{2,1}*Cov_{1,1}^{-1}*(x - ap.map.θhat_1)
# """
# function normal_expected_nuisance!(ap::AsymptoticPosterior{P}, x, i::Int = profile_ind(ap)) where P
#     @inbounds begin
#         sf = (x - ap.map.θhat[i]) / ap.map.cov[i,i]
#         for j in 1:i-1
#             ap.nuisance[j] = ap.map.θhat[j] + ap.map.cov[j,i] * sf
#         end
#         for j in i+1:P
#             ap.nuisance[j-1] = ap.map.θhat[j] + ap.map.cov[j,i] * sf
#         end
#     end
#     nothing
# end

"""
Naively assumes indepedence, and just uses the global maximum for the profile maximum.
This seems like it ought to perform poorly, but in tests it appears more robust.
"""
function naive_expected_nuisance!(ap::AsymptoticPosterior{P}, i::Int = profile_ind(ap)) where P
    @inbounds for j in 1:i-1
        ap.nuisance[j] = ap.map.θhat[j]# + ap.map.cov[j,i] * sf
    end
    @inbounds for j in i:P-1
        ap.nuisance[j] = ap.map.θhat[j+1]# + ap.map.cov[j,i] * sf
    end
end

function profilepdf(ap::AsymptoticPosterior{P,T}, ::Val{reset_search_start}=Val{true}()) where {reset_search_start,P,T}
    i = profile_ind(ap)
    # expected_nuisance!(ap, x, i)
    reset_search_start && naive_expected_nuisance!(ap, i)
    # DifferentiableObjects.initial_state!(ap.state, ap.method, ap.od, ap.nuisance)
    # I'm seeing how just using I as the initial H works out.
    # setinvH!(ap.state.invH, ap.map.cov, i) # Approximate inverse hessian with hessian at maximum.
    scaled_nlmax, scale = optimize_scale!(ap.state, ap.od, ap.nuisance, ap.method, one(T))
    ap.nlmax[] = scaled_nlmax / scale
    # copyto!(ap.nuisance, ap.state.x_old)
    # ap.nlmax[]
end
function profilepdf(ap::AsymptoticPosterior{P,T}, x, i, ::Val{reset_search_start}=Val{true}()) where {reset_search_start,P,T}
    set!(ap, x, i)
    # expected_nuisance!(ap, x, i)
    # Reseset the other parameters to the corresponding mode.
    reset_search_start && naive_expected_nuisance!(ap, i)
    scaled_nlmax, scale = optimize_scale!(ap.state, ap.od, ap.nuisance, ap.method, one(T))
    ap.nlmax[] = scaled_nlmax / scale
    # copyto!(ap.nuisance, ap.state.x_old)
    # ap.nlmax[]
end

function (ap::AsymptoticPosterior)(theta, i = profile_ind(ap))
    debug() && println("")
    debug() && @show theta, ap.rstar[], i
    debug() && @assert isnan(theta) == false
    debug() && @show rstar_p(ap, theta, i) + ap.rstar[]
    rstar_p(ap, theta, i) + ap.rstar[]
end

# Positive if x is smaller than mode, negative if bigger
function rp(ap::AsymptoticPosterior, x, i = profile_ind(ap))
    copysign( sqrt(2(ap.nlmax[] - ap.map.nlmax[])), ap.map.θhat[i] - x)
end

function rstar_p(ap::AsymptoticPosterior{P}, theta, i::Int=profile_ind(ap)) where P

    profilepdf(ap, theta, i)
    set_buffer_to_profile!(ap, i)

    setswap!(ap, i)
    hessian!(ap.map.od.config, ap.map.buffer)

    debug() && @show ap.map.buffer
    debug() && @show hessian(ap)

    r = rp(ap, theta, i)
    r + log(qb(ap, theta, i)/r)/r
end

sym(a,i) = Symbol(a, :_, i)
function profile_correction_quote(P, R, ::Type{T}) where T
    VL = min(P, jBLAS.REGISTER_SIZE ÷ sizeof(T))
    VLT = VL * sizeof(T)
    V = SIMD.Vec{VL,T}
    # q = quote @fastmath @inbounds begin end end
    # qa = q.args[2].args[3].args[3].args
    q = quote end
    qa = q.args
    iter = P ÷ VL
    push!(qa, :(ptr_Li = pointer(Li)))
    push!(qa, :(@inbounds $(sym(:vH,0)) = $V(hess[1]) ))
    push!(qa, :(@inbounds $(sym(:vG,0)) = $V(grad[1]) ))
    for r ∈ 0:iter-1 #ps = 0
        push!(qa, :($(sym(:vL,r)) = vload($V, ptr_Li + $(r*VLT)) ))
        push!(qa, :($(sym(:vHL,r)) = $(sym(:vH,0))*$(sym(:vL,r)) ))
        push!(qa, :($(sym(:vGL,r)) = $(sym(:vG,0))*$(sym(:vL,r)) ))
    end
    itermin = 0
    for pb ∈ 0:VL:P-2
        for ps ∈ max(1,pb):min( pb+VL-1, P-2 )
            push!(qa, :(@inbounds $(sym(:vH,ps)) = $V(hess[$(ps+1)])))
            push!(qa, :(@inbounds $(sym(:vG,ps)) = $V(grad[$(ps+1)])))
            for r ∈ itermin:iter-1
                push!(qa, :($(sym(:vL,r)) = vload($V, ptr_Li + $(r*VLT) + $(sizeof(T)*R*ps))))
                push!(qa, :($(sym(:vHL,r)) = fma($(sym(:vH,ps)),$(sym(:vL,r)),$(sym(:vHL,r))) ))
                push!(qa, :($(sym(:vGL,r)) = fma($(sym(:vG,ps)),$(sym(:vL,r)),$(sym(:vGL,r))) ))
            end
        end
        if itermin == 0
            push!(qa, :(Vout = $(sym(:vHL,itermin)) * $(sym(:vGL,itermin))))
        else
            push!(qa, :(Vout = fma($(sym(:vHL,itermin)), $(sym(:vGL,itermin)),Vout)))
        end
        itermin += 1
    end
    push!(qa, :(sum(Vout)))
    q
end
"""
Calculates the quadratic form of grad' * Li' * Li * hess[1:end-1,end]
"""
@generated function profile_correction(Li::SizedSIMDMatrix{Pm1,Pm1,T,R},
                        grad::SizedSIMDVector, hess::SizedSIMDMatrix) where {Pm1,T,R}
    profile_correction_quote(Pm1+1, R, T)
end

function fdf_adjrstar_p(ap::AsymptoticPosterior{P,T}, theta, p_i::Int=profile_ind(ap),
                    ::Val{reset_search_start} = Val{true}()) where {P,T,reset_search_start}


    profilepdf(ap, theta, p_i, Val{reset_search_start}())
    set_buffer_to_profile!(ap, p_i)
    setswap!(ap, p_i)
    hess = hessian!(ap.map.od.config, ap.map.buffer)

    delta_log_likelihood = ap.nlmax[]-ap.map.nlmax[]
    r = copysign( sqrt((T(2))*delta_log_likelihood), ap.map.θhat[p_i] - theta)

    grad = gradient(ap)

    Li = ap.Lsmall
    rootdet = invcholdet!(Li, hess)
    # success || return $(T(Inf),T(Inf))

    # prof_factor = $(zero(T))
    # $q
    # @inbounds for i in 1:P-1
    #     hᵢₙ = hessian(ap)[i,end]
    #     gᵢ = grad[i]
    #     for j in 1:i-1
    #         prof_factor += ap.subhess[j,i] * (grad[j]*hᵢₙ + gᵢ*hessian(ap)[j,end])
    #     end
    #     prof_factor += ap.subhess[i,i] * gᵢ * hᵢₙ
    # end
    prof_factor = profile_correction(Li, grad, hess)
    hess_adjust = rootdet * ap.map.base_adjust[]
    @inbounds q = (prof_factor - grad[P]) * hess_adjust

    rstar = r + log(q/r)/r
    rstar + ap.rstar[], exp(T(0.5)*abs2(rstar)-delta_log_likelihood) / hess_adjust
end
function pdf(ap::AsymptoticPosterior{P,T}, theta, i = profile_ind(ap),::Val{reset_search_start}=Val{true}()) where {P,T,Pm1,reset_search_start}

    profilepdf(ap, theta, i, Val{reset_search_start}())
    set_buffer_to_profile!(ap, i)
    setswap!(ap, i)
    hessian!(ap.map.od.config, ap.map.buffer)

    # is there a more efficient way of calculating the determinant?
    # skip storing? Implement that...
    rootdet, success = safecholdet!(hessian(ap), Val{Pm1}())
    success ? exp(ap.map.nlmax[]-ap.nlmax[]) / (sqrt(2π) * rootdet * ap.map.base_adjust[]) : T(Inf)
end


function qb(ap::AsymptoticPosterior{P,T}, theta, pi = profile_ind(ap)) where {P,T}
    grad = gradient(ap)
    hess = hessian(ap) # P x P
    Li = ap.Lsmall
    rootdet = invcholdet!(Li, hess)
    prof_factor = profile_correction(Li, grad, hess)

    # @inbounds for i in 1:P-1
    #     hᵢₙ = hessian(ap)[i+$(TriangularMatrices.small_triangle(P))]
    #     gᵢ = grad[i]
    #     for j in 1:i-1
    #         prof_factor += ap.subhess[j,i] * (grad[j]*hᵢₙ + gᵢ*hessian(ap)[j,end])
    #     end
    #     prof_factor += ap.subhess[i,i] * gᵢ * hᵢₙ
    # end
    # @show -prof_factor
    # @show grad[P]
    # @show (prof_factor - grad[P])
    # @show rootdet
    # @show ap.map.base_adjust[]
    @inbounds (prof_factor - grad[P]) * rootdet * ap.map.base_adjust[]
end

function set_buffer_to_profile!(ap::AsymptoticPosterior{P}, i = profile_ind(ap)) where P
    @inbounds begin # start with bounds-check=yes while testing
        for j in 1:i-1
            ap.map.buffer[j] = ap.nuisance[j]
        end
        if i != P
            ap.map.buffer[i] = ap.nuisance[P-1]
            for j in i+1:P-1
                ap.map.buffer[j] = ap.nuisance[j-1]
            end
        end
        ap.map.buffer[P] = profile_val(ap)
    end
end

Φ⁻¹(x::T) where T = √T(2)*erfinv(T(2)x-T(1))
function Statistics.quantile(ap::AsymptoticPosterior, alpha, i)
    ap.od.config.f.i[] = i
    ap.rstar[] = Φ⁻¹(alpha)
    # quadratic_search(ap, i)
    linear_search(ap, profile_ind(ap))
end
function Statistics.quantile(ap::AsymptoticPosterior, alpha)
    ap.rstar[] = Φ⁻¹(alpha)
    linear_search(ap, profile_ind(ap))
    # quadratic_search(ap, profile_ind(ap))
end


export lquantile, qquantile
function lquantile(ap::AsymptoticPosterior, alpha, i)
    ap.od.config.f.i[] = i
    ap.rstar[] = Φ⁻¹(alpha)
    linear_search(ap, i)
end
function lquantile(ap::AsymptoticPosterior, alpha)
    ap.rstar[] = Φ⁻¹(alpha)
    linear_search(ap, profile_ind(ap))
end

function qquantile(ap::AsymptoticPosterior, alpha, i)
    ap.od.config.f.i[] = i
    ap.rstar[] = Φ⁻¹(alpha)
    quadratic_search(ap, i)
end
function qquantile(ap::AsymptoticPosterior, alpha)
    ap.rstar[] = Φ⁻¹(alpha)
    quadratic_search(ap, profile_ind(ap))
end


# (ap::AsymptoticPosterior)(x) = ap.map.od(x)
mode(ap::AsymptoticPosterior) = ap.map.θhat
