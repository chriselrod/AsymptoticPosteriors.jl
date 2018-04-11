

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

