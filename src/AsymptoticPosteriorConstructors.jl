


function ProfileDifferentiable(f::F, x::PaddedMatrices.AbstractMutableFixedSizePaddedVector{Pm1,T,R,R}, ::Val{P}) where {F,T,Pm1,P,R}

    result = DifferentiableObjects.GradientDiffResult(x)
    chunk = DifferentiableObjects.Chunk(Val{Pm1}())
    gconfig = ForwardDiff.GradientConfig(nothing, x, chunk, ForwardDiff.Tag(nothing, T))
    profile = ProfileWrapper(f,
        MutableFixedSizePaddedVector{P,T}(undef),
        MutableFixedSizePaddedVector{P,eltype(gconfig)}(undef),
        Ref(P), Ref(zero(T))
    )

    gradconfig = GradientConfiguration(profile, result, gconfig)

    OnceDifferentiable(similar(x), similar(x), gradconfig)#, Val{P}())
end


@generated function PrePaddedMatrix(::Val{M}, ::Val{N}, ::Type{T}) where {M,N,T}
    # R, L = calculate_L_from_size((M,N), T)
    R = PaddedMatrices.calc_padding(M, T)
    :(zero(MutableFixedSizePaddedMatrix{$R,$N,$T})) # we want offset
end

function AsymptoticPosteriorFD(f, map_::MAPFD, initial_x::MutableFixedSizePaddedVector{P,T}) where {T,P}

    ValP = Val{P}()
    ValPm1 = DifferentiableObjects.ValM1(Val{P}())

    backtrack = DifferentiableObjects.BackTracking2(Val(2)) # order 2 backtrack vs 3
    # state = DifferentiableObjects.uninitialized_state(nuisance)
    state = DifferentiableObjects.BFGSState2(ValPm1)
    od = ProfileDifferentiable(f, DifferentiableObjects.ref_x_old(state), Val{P}())

    Lsmall = PrePaddedMatrix(ValPm1,ValPm1,T)

    AsymptoticPosteriorFD(od, backtrack, state, map_, Lsmall)
end
function AsymptoticPosteriorFD(od::D, method::M, state::S, map_::MAP, Lsmall::SM) where {P,T,Pm1,R,D<:AbstractDifferentiableObject{Pm1,T},M,S,MAP <: MAPFD{P,T,R},SV <: PaddedMatrices.AbstractMutableFixedSizePaddedVector{Pm1,T},R2,SM <: MutableFixedSizePaddedMatrix{R2,Pm1,T}}

    AsymptoticPosteriorFD{P,T,Pm1,R,D,M,S,MAP,R2,SM}(od, method, state, map_, Ref{T}(), Ref{T}(), Lsmall)#, @view(hessian(map_)[1:P-1,1:P-1]))
end

# Convenience function does a dynamic dispatch if we don't start with a MutableFixedSizePaddedVector
# If we have a static arrays dependency, we may also support static dispatches with that?
function AsymptoticPosteriorFD(f, initial_x::AbstractArray{T}) where T
    AsymptoticPosteriorFD(f, MutableFixedSizePaddedVector{length(initial_x),T}(initial_x))
end

function AsymptoticPosteriorFD(f, initial_x::MutableFixedSizePaddedVector{P,T}) where {P,T}
    map_ = MAPFD(f, initial_x)
    fit!(map_, initial_x)
    AsymptoticPosteriorFD(f, map_, initial_x)
end
function AsymptoticPosteriorFD(::UndefInitializer, f, initial_x::MutableFixedSizePaddedVector{P,T}) where {P,T}
    map_ = MAPFD(f, initial_x)
    # fit!(map_, initial_x)
    AsymptoticPosteriorFD(f, map_, initial_x)
end

