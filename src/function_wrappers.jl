
struct ProfileWrapper{N,F,T,A<:AbstractArray{T},B<:AbstractArray}#,C<:AbstractArray}
    f::F
    x::A
    y::B
    # z::C
    i::Base.RefValue{Int}
    v::Base.RefValue{T}
end
function (pw::ProfileWrapper{N,F,T,A})(x::A) where {N,F,T,A <: AbstractArray}
    debug() && @show x
    @inbounds begin
        for i in 1:pw.i[]-1
            pw.x[i] = x[i]
        end
        # pw.x[pw.i[]] = pw.v[]
        for i in pw.i[]+1:N
            pw.x[i] = x[i-1]
        end
    end
    debug() && @show pw.x
    - pw.f(pw.x)
end
function (pw::ProfileWrapper{N,F,T,A,B})(y::B) where {N,F,T,A,B <: AbstractArray}
    # @show y
    @inbounds begin
        for i in 1:pw.i[]-1
            pw.y[i] = y[i]
        end
        # pw.x[pw.i[]] = pw.v[]
        for i in pw.i[]+1:N
            pw.y[i] = y[i-1]
        end
    end
    # @show pw.y
    - pw.f(pw.y)
end
# function (pw::ProfileWrapper{N,F,T,A,B,C})(z::C) where {N,F,T,A,B,C}
#     @inbounds begin
#         for i in 1:pw.i[]-1
#             pw.z[i] = z[i]
#         end
#         # pw.x[pw.i[]] = pw.v[]
#         for i in pw.i[]+1:N
#             pw.z[i] = z[i-1]
#         end
#     end
#     - pw.f(pw.z)
# end
# function (pw::ProfileWrapper{N})(w) where N #fallback method
#     wc = Vector{eltype(w)}(undef, N)
#     @inbounds begin
#         for i in 1:pw.i[]-1
#             wc[i] = w[i]
#         end
#         wc[pw.i[]] = pw.v[]
#         for i in pw.i[]+1:N
#             wc[i] = w[i-1]
#         end
#     end
#     - pw.f(wc)
# end
function set!(pw::ProfileWrapper, v, i::Int)
    pw.v[] = v
    pw.i[] = i
    @inbounds pw.x[i] = v
    @inbounds pw.y[i] = v
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
Swap(f::F, ::Val{N}) where {F,N} = Swap{N,F}(f,Ref(N))
