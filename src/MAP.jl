
abstract type AbstractMAP{P,T,R} end

"""Here, we take a much more straightforward approach to the implementation of Reid."""
struct MAPFD{P,T,R,D<:DifferentiableObjects.AbstractDifferentiableObject{P},M,S,L} <: AbstractMAP{P,T,R}
    od::D
    method::M
    state::S
    θhat::PaddedMatrices.PtrVector{P,T,R,R}
    buffer::MutableFixedSizeVector{P,T,R,R}
    nlmax::Base.RefValue{T}
    base_adjust::Base.RefValue{T}
    std_estimates::MutableFixedSizeVector{P,T,R,R}
    Lfull::MutableFixedSizeMatrix{P,P,T,R,L}
end
@inline setswap!(map_::MAPFD, i) = map_.od.config.f.i[] = i
@inline gradient(map_::MAPFD) = map_.od.config.result.grad
@inline hessian(map_::MAPFD) = DiffResults.hessian(map_.od.config.result)

@inline Lfull(map_::MAPFD) = map_.Lfull


fit!(map_::MAPFD) = fit!(map_, map_.buffer)
function fit!(map_::MAPFD{P,T}, initial_x::AbstractFixedSizeVector{P,T}) where {P,T}
    setswap!(map_, P)
    # DifferentiableObjects.initial_state!(map_.state, map_.method, map_.od, initial_x)
    scaled_nlmax, scale = optimize_scale!(map_.state, map_.od, initial_x, map_.method, one(T))
    map_.nlmax[] = scaled_nlmax / scale
    θhat = map_.θhat

    hessian!(map_.od.config, θhat)

    # @show inv(hessian(map_))
    # @show hessian(map_)
    map_.base_adjust[] = PaddedMatrices.invcholdetLLi!(Lfull(map_), hessian(map_))
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

function MAPFD(f, initial_x::MutableFixedSizeVector{P,T}) where {T,P}

    od = swappable_twice_differentiable(f, Val(P))

    state = DifferentiableObjects.BFGSState(Val(P))
    backtrack = DifferentiableObjects.BackTracking2(Val(2))#order 2 backtrack

    Lfull = zero(MutableFixedSizeMatrix{P,P,T})

    map_ = MAPFD(od, backtrack, state, DifferentiableObjects.ref_x_old(state), similar(initial_x), Ref{T}(), Ref{T}(), PaddedMatrices.mutable_similar(initial_x), Lfull)
end
function MAPFD(od::D, method::M, state::S, θhat::MutableFixedSizeVector{P,T},
                buffer::MutableFixedSizeVector{P,T}, nlmax::Base.RefValue{T},
                base_adjust::Base.RefValue{T}, std_estimates::MutableFixedSizeVector{P,T},
                Lfull::MutableFixedSizeMatrix{P,P,T,R,L}) where {P,T,R,D<:DifferentiableObjects.AbstractDifferentiableObject{P},M,S,L}
    MAPFD{P,T,R,D,M,S,L}(od, method, state, θhat, buffer, nlmax, base_adjust, std_estimates, Lfull)
end
