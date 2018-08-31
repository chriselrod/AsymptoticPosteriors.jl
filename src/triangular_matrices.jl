

mutable struct TriangularMatrix{P,T,R,L} <: AbstractMatrix{T}
    data::NTuple{L,T}
    function TriangularMatrix{P,T,R,L}(::UndefInitializer) where {P,T,R,L}
        new()
    end
    @generated function TriangularMatrix{P,T,R}(::UndefInitializer) where {P,T}
        R, L = SIMDArrays.calculate_L_from_size((P,P))
        quote
            out = TriangularMatrix{$P,$T,$R,$L}(undef)
            @inbounds for i ∈ 2:$P, j ∈ i:$P
                out[j,i] = zero($T)
            end
            out
        end
        # :(SizedSIMDArray{$S,$T,$N,$R,$L}(undef))
    end
end

Base.size(::TriangularMatrix{P}) where P = (P,P)
@generated Base.length(::TriangularMatrix{P}) where P = P*P
@generated SIMDArrays.full_length(::TriangularMatrix{P,T,R}) where {P,T,R} = P*R

@inline function Base.getindex(A::TriangularMatrix{P,T,R}, i, j) where {P,T,R}
    @boundscheck ( i > j || i > P || j > P ) && throw(BoundsError())
    i == j && return diag(A, i)
    unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), i + (j-1)*R)
end
@inline function Base.getindex(A::TriangularMatrix{P,T,R,L}, i) where {P,T,R,L}
    @boundscheck i > L && throw(BoundsError())
    unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), i)
end
@inline function diag(A::TriangularMatrix{P,T}, i) where {P,T}
    @boundscheck i > P && throw(BoundsError())
    unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), i)
end
@inline function Base.setindex!(A::TriangularMatrix{P,T,R,L}, val, i) where {P,T,R,L}
    @boundscheck i > L && throw(BoundsError())
    unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), val, i)
end
@inline function Base.setindex!(A::TriangularMatrix{P,T,R}, val, i, j) where {P,T,R}
    @boundscheck ( i > j || i > P || j > P ) && throw(BoundsError())
    if i == j
        unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), val, i )
    else
        unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(A)), val, i + (j-1)*R )
    end
end
