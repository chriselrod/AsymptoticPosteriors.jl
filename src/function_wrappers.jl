

# Should also use FunctionWrappers.jl once that supports Julia 0.7
# to avoid excessive recompilation.


struct ProfileWrapper{P,F,T1,T2}#,C<:AbstractArray}
    f::F
    x::SizedSIMDVector{P,T1}
    y::SizedSIMDVector{P,T2}
    # z::C
    i::Base.RefValue{Int}
    v::Base.RefValue{T1}
end
function (pw::ProfileWrapper{P,F,T})(x::SizedSIMDVector{T}) where {P,F,T}
    debug() && @show x
    @inbounds begin
        for i in 1:pw.i[]-1
            pw.x[i] = x[i]
        end
        # pw.x[pw.i[]] = pw.v[]
        for i in pw.i[]+1:P
            pw.x[i] = x[i-1]
        end
    end
    debug() && @show pw.x
    - pw.f(pw.x)
end

function (pw::ProfileWrapper{P,F,T,T2})(y::SizedSIMDVector{Pm1,T2}) where {P,Pm1,F,T,T2}
    # @show y
    @inbounds begin
        for i in 1:pw.i[]-1
            pw.y[i] = y[i]
        end
        # pw.x[pw.i[]] = pw.v[]
        for i in pw.i[]+1:P
            pw.y[i] = y[i-1]
        end
    end
    # @show pw.y
    - pw.f(pw.y)
end

function set!(pw::ProfileWrapper, v, i::Int)
    pw.v[] = v
    pw.i[] = i
    @inbounds pw.x[i] = v
    @inbounds pw.y[i] = v
    pw
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
