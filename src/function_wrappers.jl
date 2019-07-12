

# Probably use FunctionWrappers.jl for interactive mode, to avoid
# excessive recompilation.


struct ProfileWrapper{P,F,T1,T2,R1,R2}#,C<:AbstractArray}
    f::F
    x::MutableFixedSizePaddedVector{P,T1,R1,R1}
    y::MutableFixedSizePaddedVector{P,T2,R2,R2}
    # z::C
    i::Base.RefValue{Int}
    v::Base.RefValue{T1}
end
function (pw::ProfileWrapper{P,F,T})(x::PaddedMatrices.AbstractMutableFixedSizePaddedVector{Pm1,T}) where {P,Pm1,F,T}
    debug() && @show x
    @inbounds begin
        for i in 1:pw.i[]-1
            pw.x[i] = x[i]
        end
        # pw.x[pw.i[]] = pw.v[]
        for i in pw.i[]:Pm1
            pw.x[i+1] = x[i]
        end
    end
    debug() && @show pw.x
    - pw.f(pw.x)
end

function (pw::ProfileWrapper{P,F,T,T2})(y::PaddedMatrices.AbstractMutableFixedSizePaddedVector{Pm1,T2}) where {P,Pm1,F,T,T2}
    # @show y
    @inbounds begin
        for i in 1:pw.i[]-1
            pw.y[i] = y[i]
        end
        # pw.x[pw.i[]] = pw.v[]
        for i in pw.i[]:Pm1
            pw.y[i+1] = y[i]
        end
    end
    # @show pw.y
    - pw.f(pw.y)
end

function set_profile_val_ind!(pw::ProfileWrapper, v, i::Int)
    pw.v[] = v
    pw.i[] = i
    @inbounds pw.x[i] = v
    @inbounds pw.y[i] = v
    nothing
end

struct Swap{P,F}
    f::F
    i::Base.RefValue{Int}
end
function (s::Swap{P})(x) where P
    @inbounds x[s.i[]], x[P] = x[P], x[s.i[]]
    - s.f(x)
end
Swap(f::F, ::Val{P}) where {F,P} = Swap{P,F}(f,Ref(P))

function swappable_twice_differentiable(f, ::Val{P}) where {P}
    TwiceDifferentiable(Swap(f, Val{P}()), Val{P}())
end

# struct Swappable{P,D <: DifferentiableObjects.AbstractTwiceDifferentiableObject{P}}
#     obj::D
#     i::Base.RefValue{Int}
# end


