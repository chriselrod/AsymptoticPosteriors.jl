

# LinearAlgebra.LAPACK.potrf!('U', x) #Only good for LinearAlgebra.BLAS.BlasFloat
# LinearAlgebra.LAPACK.trtri!('U', 'N', x)


function chol!(A::AbstractMatrix, ::Type{UpperTriangular}, ::Val{N}) where N
    @inbounds @fastmath begin
        for k = 1:N
            for i = 1:k - 1
                A[k,k] -= A[i,k]'A[i,k]
            end
            Akk = chol!(A[k,k], UpperTriangular)
            A[k,k] = Akk
            AkkInv = inv(Akk') # inv(copy(Akk')) # inv isn;t in place, why copy?
            for j = k + 1:N
                for i = 1:k - 1
                    A[k,j] -= A[i,k]'A[i,j]
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
    end
    A
end
function choldet!(A::AbstractMatrix{T}, ::Type{UpperTriangular}, ::Val{N}) where {N,T}
    out = one(T)
    @inbounds @fastmath begin
        for k = 1:N
            for i = 1:k - 1
                A[k,k] -= A[i,k]'A[i,k]
            end
            Akk = chol!(A[k,k], UpperTriangular)
            A[k,k] = Akk
            out *= Akk
            AkkInv = inv(Akk') # inv(copy(Akk')) # inv isn't in place, why copy? Transpose != problem?
            for j = k + 1:N
                for i = 1:k - 1
                    A[k,j] -= A[i,k]'A[i,j]
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
    end
    out
end
function _chol!(U::AbstractMatrix, ::Val{N}, ::Type{UpperTriangular}) where N
    @inbounds @fastmath for i in 1:N
        Uii = U[i,i]
        for j in 1:i-1
            Uji = U[j,i]
            for k in 1:j-1
                Uji -= U[k,i] * U[k,j]
            end
            U[j,i] = Uji / U[j,j]
            Uii -= abs2(U[j,i])
        end
        U[i,i] = √Uii
    end
end
@inline chol!(x::Number, ::Type{T}) where T <: Union{LowerTriangular,UpperTriangular} = sqrt(x)
function chol!(A::AbstractMatrix, ::Type{LowerTriangular}, ::Val{N}) where N
    @inbounds @fastmath begin
        for k = 1:N
            for i = 1:k - 1
                A[k,k] -= A[k,i]*A[k,i]'
            end
            Akk = chol!(A[k,k], LowerTriangular) # sqrt(abs(A[k,k]))
            A[k,k] = Akk
            AkkInv = inv(Akk)
            for j = 1:k - 1
                @simd for i = k + 1:N
                    A[i,k] -= A[i,j]*A[k,j]'
                end
            end
            for i = k + 1:N
                A[i,k] *= AkkInv'
            end
        end    
    end
    A
end
function choldet!(A::AbstractMatrix{T}, ::Type{LowerTriangular}, ::Val{N}) where {N,T}
    out = one(T)
    @inbounds @fastmath begin
        for k = 1:N
            for i = 1:k - 1
                A[k,k] -= A[k,i]*A[k,i]'
            end
            Akk = chol!(A[k,k], LowerTriangular) # sqrt(abs(A[k,k]))
            A[k,k] = Akk
            out *= Akk
            AkkInv = inv(Akk)
            for j = 1:k - 1
                @simd for i = k + 1:N
                    A[i,k] -= A[i,j]*A[k,j]'
                end
            end
            for i = k + 1:N
                A[i,k] *= AkkInv'
            end
        end    
    end
    out
end

function chol!(U::AbstractMatrix{T}) where T
    N = size(U,1)
    @boundscheck @assert N == size(U,2)
    if isa(U, Matrix{<:LinearAlgebra.BLAS.BlasFloat}) && N > 15
        LinearAlgebra.LAPACK.potrf!('U', U)
        return nothing
    end
    @inbounds for i ∈ 1:N
        Uii = U[i,i]
        for j ∈ 1:i-1
            Uji = U[j,i]
            for k ∈ 1:j-1
                Uji -= U[k,i] * U[k,j]
            end
            U[j,i] = Uji / U[j,j]
            Uii -= abs2(U[j,i])
        end
        U[i,i] = √Uii
    end
end
function inv!(U::AbstractMatrix, ::Val{N}, ::Type{UpperTriangular}) where N
    @inbounds @fastmath for i ∈ 1:N
        U[i,i] = 1 / U[i,i]
        for j ∈ i+1:N
            Uij = U[i,j] * U[i,i]
            for k ∈ i+1:j-1
                Uij += U[k,j] * U[i,k]
            end
            U[i,j] = -Uij / U[j,j]
        end
    end
end
function inv!(L::AbstractMatrix, ::Val{N}, ::Type{LowerTriangular}) where N
    @inbounds @fastmath for i ∈ 1:N
        L[i,i] = 1 / L[i,i]
        for j ∈ i+1:N
            Lji = L[j,i] * L[i,i]
            for k ∈ i+1:j-1
                Lji += L[j,k] * L[k,i]
            end
            L[j,i] = -Lji / L[j,j]
        end
    end
end

function inv!(U::AbstractMatrix)
    N = size(U,1)
    @boundscheck @assert N == size(U,2)
    if isa(U, Matrix{<:LinearAlgebra.BLAS.BlasFloat}) && N > 15
        LinearAlgebra.LAPACK.trtri!('U', 'N', U)
        return nothing
    end
    @inbounds for i ∈ 1:N
        U[i,i] = 1 / U[i,i]
        for j ∈ i+1:N
            Uij = U[i,j] * U[i,i]
            for k ∈ i+1:j-1
                Uij += U[k,j] * U[i,k]
            end
            U[i,j] = -Uij / U[j,j]
        end
    end
end


function chol!(U::AbstractMatrix, Σ::AbstractMatrix)
    @inbounds for i ∈ 1:size(U,1)
        Uii = Σ[i,i]
        for j ∈ 1:i-1
            Uji = Σ[j,i]
            for k ∈ 1:j-1
                Uji -= U[k,i] * U[k,j]
            end
            U[j,i] = Uji / U[j,j]
            Uii -= abs2(U[j,i])
        end
        U[i,i] = √Uii
    end
end

function inv!(U_inverse::AbstractMatrix, U::AbstractMatrix)
    @inbounds for i ∈ 1:size(U,1)
        U_inverse[i,i] = 1 / U[i,i]
        for j ∈ i+1:size(U,1)
            Ui_ij = U[i,j] * U_inverse[i,i]
            for k ∈ i+1:j-1
                Ui_ij += U[k,j] * U_inverse[i,k]
            end
            U_inverse[i,j] = - Ui_ij / U[j,j]
        end
    end
end

"""
Returns:
(1) the square root of the determinant of the nuisance parameter block of the hessian |j_{λ,λ}|^{1/2}
(2) 
"""
function profile_hessian!(A::AbstractMatrix{T}, ::Val{N}) where {N,T}
    out = one(T)
    prof_info_uncorrected = A[end]
    @inbounds @fastmath begin
        for k = 1:N-1
            for i = 1:k - 1
                A[k,k] -= A[i,k]'A[i,k]
            end
            Akk = chol!(A[k,k], UpperTriangular)
            A[k,k] = Akk
            out *= Akk
            AkkInv = inv(Akk') # inv(copy(Akk')) # inv isn't in place, why copy? Transpose != problem?
            for j = k + 1:N
                for i = 1:k - 1
                    A[k,j] -= A[i,k]'A[i,j]
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
        for i = 1:N - 1
            A[N,N] -= A[i,N]'A[i,N]
        end
        prof = A[N,N]
        Akk = chol!(A[N,N], UpperTriangular)
        A[N,N] = Akk
        AkkInv = inv(Akk') # inv(copy(Akk')) # inv isn't in place, why copy? Transpose != problem?
    end
    out, prof, prof_info_uncorrected
end



function submat(A::AbstractMatrix{T}, k) where T
    n = size(A,1)
    out = Matrix{T}(undef, n-1,n-1)
    for i ∈ 1:k-1
        for j ∈ 1:k-1
            out[j,i] = A[j,i]
        end
        for j ∈ 1+k:n
            out[j-1,i] = A[j,i]
        end
    end
    for i ∈ k+1:n
        for j ∈ 1:k-1
            out[j,i-1] = A[j,i]
        end
        for j ∈ 1+k:n
            out[j-1,i-1] = A[j,i]
        end
    end
    out
end

@generated function estimate_stds(A::AbstractMatrix{T}, ::Val{N}) where {T,N}
    quote
        out = Vector{T}(undef, $N)
        # @fastmath @inbounds begin
        @inbounds begin
            # Base.Cartesian.@nexprs $N i -> (out_i = zero(T))
            Base.Cartesian.@nexprs $N i -> begin
                Base.Cartesian.@nexprs i-1 j -> begin
                    out_j += A[j,i]*A[j,i]
                end
                out_i = A[i,i]*A[i,i]
            end
            Base.Cartesian.@nexprs $N i -> (out[i] = sqrt(out_i))
        end
        out
    end
end


# julia> det(xx).*diag(inv(xx))
# 4-element Array{Float64,1}:
#  34.9658
#  44.7261
#  66.1783
#  67.9918

# julia> det.(submat.((xx,),1:4))
# 4-element Array{Float64,1}:
#  34.9658
#  44.7261
#  66.1783
#  67.9918

# julia> det(xx) .* abs2.(estimate_stds(inv(chol(xx)),Val(4)))
# 4-element Array{Float64,1}:
#  34.9658
#  44.7261
#  66.1783
#  67.9918
