

@inline beta_lpdf(x, alpha, beta) = (alpha-1)*log(x) + (beta-1)*log(1-x)

"""Log Density of the No Gold-Standard Model"""
function NoGoldStandard(priordata, S₁::T, S₂::T, C₁::T, C₂::T, π::Vararg{T,K}) where {T,K}
    @unpack n, απ, βπ, αS1, βS1, αS2, βS2,
        αC1, βC1, αC2, βC2 = priordata
    target = beta_lpdf(S₁, αS1, βS1) + beta_lpdf(S₂, αS2, βS2)
    target += beta_lpdf(C₁, αC1, βC1) + beta_lpdf(C₂, αC2, βC2)
    omS₁, omS₂, omC₁, omC₂ = 1-S₁, 1-S₂, 1-C₁, 1-C₂
    πₛ₁, πc₁ = S₁*S₂, C₁*C₂
    πₛ₂, πₛ₃ = S₁ - πₛ₁, S₂ - πₛ₁
    πc₂, πc₃ = C₁ - πc₁, C₂ - πc₁
    πₛ₄, πc₄ = omS₁ - πₛ₃, omC₁ - πc₃
    @inbounds for k in 1:K
        πₖ = π[k]
        omπₖ = 1 - πₖ
        target += beta_lpdf( πₖ, απ, βπ )

        target += n[1,k] * log( πₖ*πₛ₁ + omπₖ*πc₄ )
        target += n[2,k] * log( πₖ*πₛ₂ + omπₖ*πc₃ )
        target += n[3,k] * log( πₖ*πₛ₃ + omπₖ*πc₂ )
        target += n[4,k] * log( πₖ*πₛ₄ + omπₖ*πc₁ )

        target += n[5,k] * log( πₖ*  S₁ + omπₖ*omC₁ )
        target += n[6,k] * log( πₖ*omS₁ + omπₖ*  C₁ )

        target += n[7,k] * log( πₖ*  S₂ + omπₖ*omC₂ )
        target += n[8,k] * log( πₖ*omS₂ + omπₖ*  C₂ )
    end
    target
end



"""Log Density of the No Gold-Standard Model, with correlations on log-odds scale"""
function NoGoldStandardCorr(priordata, S₁::T, S₂::T, C₁::T, C₂::T, ρₛ::T, ρc::T, π::Vararg{T,K}) where {T,K}
    @unpack n, απ, βπ, αS1, βS1, αS2, βS2, αC1, βC1, αC2, βC2 = priordata
    target = beta_lpdf(S₁, αS1, βS1) + beta_lpdf(S₂, αS2, βS2) + beta_lpdf(C₁, αC1, βC1) + beta_lpdf(C₂, αC2, βC2) - 4ρₛ - 4ρc#exponential priors.
    
    omS₁, omS₂, omC₁, omC₂ = 1-S₁, 1-S₂, 1-C₁, 1-C₂
    S₁S₂ = S₁*S₂
    Sₐ = S₁ + S₂ - S₁S₂
    Sₒ = Sₐ - S₁S₂
    expnρₛ = exp(-ρₛ)
    Sdenom = 1 / (1 - Sₒ + Sₒ * expnρₛ)
    πₛ₁ = S₁S₂ * Sdenom
    πₛ₂ = (S₁-S₁S₂) * expnρₛ * Sdenom
    πₛ₃ = (S₂-S₁S₂) * expnρₛ * Sdenom
    πₛ₄ = (1 - Sₐ) * Sdenom
    C₁C₂ = C₁*C₂
    Cₐ = C₁ + C₂ - C₁C₂
    Cₒ = Cₐ - C₁C₂
    expnρc = exp(-ρc)
    Cdenom = 1 / (1 - Cₒ + Cₒ * expnρc)
    πc₁ = C₁C₂ * Cdenom
    πc₂ = (C₁-C₁C₂) * expnρc * Cdenom
    πc₃ = (C₂-C₁C₂) * expnρc * Cdenom
    πc₄ = (1 - Cₐ) * Cdenom

    @inbounds for k in 1:K
        πₖ = π[k]
        omπₖ = 1 - πₖ
        target += beta_lpdf( πₖ, απ, βπ )

        target += n[1,k] * log( πₖ*πₛ₁ + omπₖ*πc₄ )
        target += n[2,k] * log( πₖ*πₛ₂ + omπₖ*πc₃ )
        target += n[3,k] * log( πₖ*πₛ₃ + omπₖ*πc₂ )
        target += n[4,k] * log( πₖ*πₛ₄ + omπₖ*πc₁ )

        target += n[5,k] * log( πₖ*  S₁ + omπₖ*omC₁ )
        target += n[6,k] * log( πₖ*omS₁ + omπₖ*  C₁ )

        target += n[7,k] * log( πₖ*  S₂ + omπₖ*omC₂ )
        target += n[8,k] * log( πₖ*omS₂ + omπₖ*  C₂ )
    end

    target
end

@generated function NGS(data, params::AbstractVector{T}, ::Val{N}) where {N,T}
    call = generate_call_transform(:NoGoldStandard, N )
#    Nm4 = N - 4
    quote
        @nexprs $N i -> begin
            @inbounds p_i = inv_logit(params[i])
        end
        $call
        target
    end
end



@inline logit(x) = log(x/(1-x))
@inline inv_logit(x) = 1/(1+exp(-x))
















struct Multinomial{N,T}
    n::Int
    p::NTuple{N,T}
    cumulative_p::NTuple{N,T}
end
function Multinomial(n, p::NTuple{N,T}) where {N,T}
    total = Ref{T}(zero(T)) #Cumsum not defined for ntuples; this gets around closure bug.
    cumulative_p = ntuple( i -> begin
        total[] += p[i]
    end, Val{N}())
    Multinomial(n, p, cumulative_p)
end
function Random.rand(m::Multinomial{N,T}, n::Int...) where {N,T}
    rand!(Array{NTuple{N,Int}}(undef, n...), m)
end
@generated function Random.rand!(out::AbstractArray{NTuple{N,I}}, m::Multinomial{N,T}, rng = Compat.Random.GLOBAL_RNG) where {N,T,I<:Integer}
    quote
        for i ∈ eachindex(out)
            @nexprs $N d -> x_d = zero(I)
            for j ∈ 1:m.n
                r = rand(rng)
                @nif $N d -> r < m.cumulative_p[d] d -> x_d += one(I)
            end
            out[i] = rand(m)
        end
        out
    end
end
@generated function Random.rand(m::Multinomial{N,T}, rng = Compat.Random.GLOBAL_RNG) where {N,T}
    quote
        @nexprs $N d -> x_d = 0
        for j ∈ 1:m.n
            r = rand(rng)
            @nif $N d -> r < m.cumulative_p[d] d -> x_d += 1
        end
        @ntuple $N d -> x_d
    end
end
# struct CorrErrors{p, T<: Real}
#     π::SVector{p,T}
#     S::SVector{2,T}
#     C::SVector{2,T}
#     covsc::SVector{2,T}
# end
# function CorrErrors(π::NTuple{p,T}, S::Tuple{T,T}, C::Tuple{T,T}, covsc::Tuple{T,T} = (0.0,0.0)) where {p,T}
#     CorrErrors{p,T}(SVector(π), SVector(S), SVector(C), SVector(covsc))
# end
struct CorrErrors{P,T<: Real,PP}
    π::NTuple{P,T}
    S::NTuple{2,T}
    C::NTuple{2,T}
    covsc::NTuple{2,T}
    p_stick::NTuple{P,NTuple{3,T}}
    p_stick_independent::NTuple{P,NTuple{3,T}}
    p_marginal1::NTuple{P,T}
    p_marginal2::NTuple{P,T}
    Θ::NTuple{PP,T}
end
function CorrErrors(π::NTuple{P,T}, S::Tuple{T,T}, C::Tuple{T,T}, covsc::Tuple{T,T} = (0.0,0.0)) where {P,T}
    p = common_p(π, S, C, covsc)
    p_ind = common_p(π, S, C, (0.0,0.0))
    CorrErrors(π, S, C, covsc, break_sticks.(p), break_sticks.(p_ind), p_i(π, S, C, 1), p_i(π, S, C, 2), unconstrain(π, S, C, covsc))
end
# function CorrErrors
@generated function unconstrain(π::NTuple{P}, S::NTuple{2}, C::NTuple{2}, covsc::NTuple{2}) where P
    PP = P+6
    quote
        x_1 = logit(2S[1]-1)
        x_2 = logit(2S[2]-1)
        x_3 = logit(2C[1]-1)
        x_4 = logit(2C[2]-1)
        x_5 = log(covsc[1])
        x_6 = log(covsc[2])
        @nexprs $P i -> begin
            x_{i+6} = logit(π[i])
        end
        @ntuple $PP i -> x_i
    end
end
function common_p(Θ::CorrErrors{P,Float64}) where P
    @unpack π, S, C, covsc = Θ
    common_p(π, S, C, covsc)
end
function common_p(π::NTuple{P}, S::NTuple{2}, C::NTuple{2}, covsc::NTuple{2}) where P
    omπ = 1 .- π
    S₁, S₂ = S
    C₁, C₂ = C
    ρₛ, ρc = covsc
    omS₁, omS₂, omC₁, omC₂ = 1-S₁, 1-S₂, 1-C₁, 1-C₂
    S₁S₂ = S₁*S₂
    Sₐ = S₁ + S₂ - S₁S₂
    Sₒ = Sₐ - S₁S₂
    expnρₛ = exp(-ρₛ)
    Sdenom = 1 / (1 - Sₒ + Sₒ * expnρₛ)
    πₛ₁ = S₁S₂ * Sdenom
    πₛ₂ = (S₁-S₁S₂) * expnρₛ * Sdenom
    πₛ₃ = (S₂-S₁S₂) * expnρₛ * Sdenom
    πₛ₄ = (1 - Sₐ) * Sdenom
    C₁C₂ = C₁*C₂
    Cₐ = C₁ + C₂ - C₁C₂
    Cₒ = Cₐ - C₁C₂
    expnρc = exp(-ρc)
    Cdenom = 1 / (1 - Cₒ + Cₒ * expnρc)
    πc₁ = C₁C₂ * Cdenom
    πc₂ = (C₁-C₁C₂) * expnρc * Cdenom
    πc₃ = (C₂-C₁C₂) * expnρc * Cdenom
    πc₄ = (1 - Cₐ) * Cdenom
  ntuple( i -> (
      π[i]*πₛ₁ + omπ[i]*πc₄,
      π[i]*πₛ₂ + omπ[i]*πc₃,
      π[i]*πₛ₃ + omπ[i]*πc₂,
      π[i]*πₛ₄ + omπ[i]*πc₁
  ), Val{P}())
end
function p_i(Θ::CorrErrors, i::Int)
    @unpack π, S, C = Θ
    p_i(π, S, C, i)
end
function p_i(π::NTuple{P}, S::NTuple{2}, C::NTuple{2}, i) where P
    ntuple(j -> π[j]*S[i] + (1-π[j])*(1-C[i]), Val{P}())
end
function p_i2(Θ::CorrErrors, i::Int)
    @unpack π, S, C = Θ
    p_i2(π, S, C, i)
end
function p_i2(π::NTuple{P}, S::NTuple{2}, C::NTuple{2}, i) where P
  [(π[j]*S[i] + (1-π[j])*(1-C[i]), π[j]*(1-S[i]) + (1-π[j])*C[i]) for j ∈ 1:P]
end
@generated ValM1(::Val{N}) where N = Val{N-1}()
@generated function break_sticks(tup::NTuple{N,T}) where {N,T}
    Nm2 = N-2
    Nm1 = N-1
    quote
        p_0 = tup[1]
        cumulative_complete = one(T)
        @nexprs $Nm2 d -> begin
            cumulative_complete -= tup[d]
            p_d = tup[d+1] / cumulative_complete
        end
        @ntuple $Nm1 d -> p_{d-1}
    end
end

# gen_data(Θ::CorrErrors{P}, n_common, n_1_only, n_2_only) where P = gen_data!(Matrix{Int}(undef, 8, P), Θ, n_common, n_1_only, n_2_only)
# function gen_data!(out::Matrix{Int}, Θ::CorrErrors{P,T}, n_common::Int, n_1_only::Int, n_2_only::Int) where {P,T}
#     double_test = common_p(Θ)
#     p_1_only = p_i2(Θ, 1)
#     p_2_only = p_i2(Θ, 2)
#     for i ∈ 1:P
#         out[1:4,i] .= rand(Multinomial(n_common, double_test[i]))
#         out[5:6,i] .= rand(Multinomial(n_1_only, p_1_only[i]))
#         out[7:8,i] .= rand(Multinomial(n_2_only, p_2_only[i]))
#     end
#     out
# end


gen_data(Θ::CorrErrors{P}, n_common, n_1_only, n_2_only) where P = gen_data!(Matrix{Int}(undef, 8, P), Θ, n_common, n_1_only, n_2_only)
function gen_data!(out::Matrix{Int}, Θ::CorrErrors{P,T}, n_common::Int, n_1_only::Int, n_2_only::Int) where {P,T}
    for i ∈ 1:P
        cum_p = zero(T)
        x = rbinom(1, n_common, Θ.p_stick[i][1])[1]
        out[1,i] = x
        n = n_common - x
        x = rbinom(1, n, Θ.p_stick[i][2])[1]
        out[2,i] = x
        n -= x
        x = rbinom(1, n, Θ.p_stick[i][3])[1]
        out[3,i] = x
        out[4,i] = n - x
        x = rbinom(1, n_1_only, Θ.p_marginal1[i])[1]
        out[5,i] = x
        out[6,i] = n_1_only -x
        x = rbinom(1, n_2_only, Θ.p_marginal2[i])[1]
        out[7,i] = x
        out[8,i] = n_2_only - x
    end
    out
end
function gen_data!(out::Matrix{Int}, Θ::CorrErrors{P,T}, n_common::NTuple{P,Int}, n_1_only::NTuple{P,Int}, n_2_only::NTuple{P,Int}) where {P,T}
    for p ∈ 1:P
        cum_p = zero(T)
        x = rbinom(1, n_common[p], Θ.p_stick[p][1])[1]
        out[1,p] = x
        n = n_common[p] - x
        x = rbinom(1, n, Θ.p_stick[p][2])[1]
        out[2,p] = x
        n -= x
        x = rbinom(1, n, Θ.p_stick[p][3])[1]
        out[3,p] = x
        out[4,p] = n - x
        x = rbinom(1, n_1_only[p], Θ.p_marginal1[p])[1]
        out[5,p] = x
        out[6,p] = n_1_only[p] - x
        x = rbinom(1, n_2_only[p], Θ.p_marginal2[p])[1]
        out[7,p] = x
        out[8,p] = n_2_only[p] - x
    end
    out
end
# corr_errors_independent2 = CorrErrors((0.1,        0.4), (0.9, 0.95), (0.85,0.97));
# corr_errors_independent3 = CorrErrors((0.1,  0.25, 0.4), (0.9, 0.95), (0.85,0.97));
# corr_errors_independent4 = CorrErrors((0.1,0.2,0.3,0.4), (0.9, 0.95), (0.85,0.97));

# init6 = logit.([0.9*2-1, 0.95*2-1, 0.85*2-1, 0.97*2-1, 0.1, 0.4]);
# init7 = logit.([0.9*2-1, 0.95*2-1, 0.85*2-1, 0.97*2-1, 0.1, 0.25, 0.4]);
# init8 = logit.([0.9*2-1, 0.95*2-1, 0.85*2-1, 0.97*2-1, 0.1, 0.2, 0.3, 0.4]);
# initCorr8 = logit.([0.9*2-1, 0.95*2-1, 0.85*2-1, 0.97*2-1, 0.5/1.5, 0.45/1.45, 0.1, 0.4]);
# initCorr9 = logit.([0.9*2-1, 0.95*2-1, 0.85*2-1, 0.97*2-1, 0.5/1.5, 0.45/1.45, 0.1, 0.25, 0.4]);
# initCorr10 = logit.([0.9*2-1, 0.95*2-1, 0.85*2-1, 0.97*2-1, 0.5/1.5, 0.45/1.45, 0.1, 0.2, 0.3, 0.4]);




# @with_kw struct NoGoldData
#     απ::Float64 = 1.0
#     βπ::Float64 = 1.0
#     αS1::Float64 = 1.0
#     βS1::Float64 = 1.0
#     αS2::Float64 = 1.0
#     βS2,::Float64 = 1.0
#     αC1::Float64 = 1.0
#     βC1::Float64 = 1.0
#     αC2::Float64 = 1.0
#     βC2::Float64 = 1.0
#     n::Matrix{Int}
# end

abstract type NGD{N} end
@with_kw struct NoGoldData{N} <: NGD{N}
    n::Matrix{Int}
    απ::Float64 = 1.0 # απ
    βπ::Float64 = 1.0 # βπ
    αS1::Float64 = 1.0 # αS1
    βS1::Float64 = 1.0 # βS1
    αS2::Float64 = 1.0 # αS2
    βS2::Float64 = 1.0 # βS2
    αC1::Float64 = 1.0 # αC1
    βC1::Float64 = 1.0 # βC1
    αC2::Float64 = 1.0 # αC2
    βC2::Float64 = 1.0 # βC2
end
@with_kw struct NoGoldDataCorr{N} <: NGD{N}
    n::Matrix{Int}
    απ::Float64 = 1.0 # απ
    βπ::Float64 = 1.0 # βπ
    αS1::Float64 = 38.0 # αS1
    βS1::Float64 = 2.0 # βS1
    αS2::Float64 = 38.0 # αS2
    βS2::Float64 = 2.0 # βS2
    αC1::Float64 = 38.0 # αC1
    βC1::Float64 = 2.0 # βC1
    αC2::Float64 = 38.0 # αC2
    βC2::Float64 = 2.0 # βC2
end


function generate_call_transform(fname, n)
    # verind = VERSION > v"0.6.9" ? 3 : 2
    out = quote
        target = $fname( data, (p_1+1)/2 )
        target += log(p_1) + log(1 - p_1)
        target += log(p_5) + log(1 - p_5)
        target
    end
    for i ∈ 2:4
        p_i = Symbol(:p_, i)
        push!(out.args[2].args[2].args, :( ($p_i+1)/2 ))
        push!(out.args[4].args[2].args, :( log($p_i) ))
        push!(out.args[4].args[2].args, :( log(1-$p_i)  ))
    end
    push!(out.args[2].args[2].args, :(p_5))
    for i ∈ 6:n
        p_i = Symbol(:p_, i)
        push!(out.args[2].args[2].args, :($p_i))
        push!(out.args[6].args[2].args, :( log($p_i) ))
        push!(out.args[6].args[2].args, :( log(1-$p_i)  ))
    end
    out
end
@generated function NGS(data::NoGoldData{N}, params::AbstractVector{T}) where {N,T}
    call = generate_call_transform(:NoGoldStandard, N )
#    Nm4 = N - 4
    quote
        @nexprs $N i -> begin
            @inbounds p_i = inv_logit(params[i])
        end
        $call
        target
    end
end


function generate_call_transformc(fname, n)
    out = quote
        target = 2p_5 + 2p_6 + log(p_1) + log(1 - p_1)  #2p_5 and 2p_6 because Gamma(2,4) prior + Jacobian (+p_5+p_6).
        p_1, p_2, p_3, p_4 = (p_1+1)/2, (p_2+1)/2, (p_3+1)/2, (p_4+1)/2
        target += $fname( data, p_1 )
        target += log(p_7) + log(1 - p_7)
        #target += log(p_5) + log(1 - p_5) + log(p_6) + log(1 - p_6)
        target
    end
    for i ∈ 2:4
        p_i = Symbol(:p_, i)
        push!(out.args[6].args[2].args, :( $p_i ))
        push!(out.args[2].args[2].args, :( log($p_i)  ))
        push!(out.args[2].args[2].args, :( log(1-$p_i)  ))
    end
    push!(out.args[6].args[2].args, :(exp(p_5)))
    push!(out.args[6].args[2].args, :(exp(p_6)))
    push!(out.args[6].args[2].args, :p_7)
    for i ∈ 8:n
        p_i = Symbol(:p_, i)
        push!(out.args[6].args[2].args, :($p_i))
        push!(out.args[8].args[2].args, :( log($p_i) ))
        push!(out.args[8].args[2].args, :( log(1-$p_i)  ))
    end
    out
end
@generated function NGSC(data::NoGoldDataCorr{N}, params::AbstractVector{T}) where {N,T}
    call = generate_call_transformc(:NoGoldStandardCorr, N )
    Nm6 = N - 6
    quote
        @inbounds  begin
            @nexprs 4 i -> begin
                p_i = inv_logit(params[i])
            end
            p_5 = params[5]
            p_6 = params[6]
            @nexprs $Nm6 i -> begin
                p_{i+6} = inv_logit(params[i+6])
            end
        end
        $call
        target
    end
end
(data::NoGoldData{N})(x) where N = NGS(data, x)
(data::NoGoldDataCorr{N})(x) where N = NGSC(data, x)


@generated ValP6(::Val{N}) where N = Val{N+6}()
@generated ValP4(::Val{N}) where N = Val{N+4}()
@generated function NoGoldData(Θ::CorrErrors{P,T}, n_common::Int, n_1_only::Int, n_2_only::Int; kwargs...) where {P,T}
    :(NoGoldData{$(P+4)}(n = gen_data(Θ,  n_common,  n_1_only,  n_2_only), kwargs...))
end
@generated function NoGoldData(Θ::CorrErrors{P,T}; kwargs...) where {P,T}
    :(NoGoldData{$(P+4)}(n = Matrix{Int}(undef,8,$P), kwargs...))
end
function NoGoldData(Θ::CorrErrors{P}, n_common::Int, n_1_only::Int, n_2_only::Int; kwargs...) where P
    NoGoldData(ValP4(Val{P}()); n = gen_data(Θ,  n_common,  n_1_only,  n_2_only), kwargs...)
end
function NoGoldData(::Val{N}; kwargs...) where {N}
    NoGoldData{N}(; kwargs...)
end
function NoGoldDataCorr(Θ::CorrErrors{P}, n_common::Int, n_1_only::Int, n_2_only::Int; kwargs...) where P
    NoGoldDataCorr(ValP6(Val{P}()); n = gen_data(Θ,  n_common,  n_1_only,  n_2_only), kwargs...)
end
function NoGoldDataCorr(::Val{N}; kwargs...) where {N}
    NoGoldDataCorr{N}(; kwargs...)
end

function resample!(data::NGD{N}, Θ, n_common, n_1_only, n_2_only) where N
    gen_data!(data.n, Θ, n_common, n_1_only, n_2_only)
end

# n_small_2  = NoGoldData(corr_errors_independent2,  100,  200,  16);
# n_medium_2 = NoGoldData(corr_errors_independent2,  400,  800,  64);
# n_big_2    = NoGoldData(corr_errors_independent2, 1600, 3200, 252);

# n_small_3  = NoGoldData(corr_errors_independent3,  100,  200,  16);
# n_medium_3 = NoGoldData(corr_errors_independent3,  400,  800,  64);
# n_big_3    = NoGoldData(corr_errors_independent3, 1600, 3200, 252);

# n_small_4  = NoGoldData(corr_errors_independent4,  100,  200,  16);
# n_medium_4 = NoGoldData(corr_errors_independent4,  400,  800,  64);
# n_big_4    = NoGoldData(corr_errors_independent4, 1600, 3200, 252);

# zeros6 = fill(0.0,6);
# zeros7 = fill(0.0,7);
# zeros8 = fill(0.0,8);
# zeros9 = fill(0.0,9);
# zeros10 = fill(0.0,10);

function inverse_transform(x)
    out = similar(x)
    @inbounds for i ∈ 1:4
        out[i] = (inv_logit(x[i]) + 1) / 2
    end
    @inbounds for i ∈ 5:length(x)
        out[i] = inv_logit(x[i])
    end
    out
end
function inverse_transformc(x)
    out = similar(x)
    @inbounds for i ∈ 1:4
        out[i] = (inv_logit(x[i]) + 1) / 2
    end
    @inbounds for i ∈ 5:6
        out[i] = exp(x[i])
    end
    @inbounds for i ∈ 7:length(x)
        out[i] = inv_logit(x[i])
    end
    out
end

interval(ap, alpha, ind) = (quantile(ap,alpha/2,ind), quantile(ap,1-alpha/2,ind))
four_num_sum(ap, ind) = (quantile(ap,0.025,ind), quantile(ap,0.25,ind), quantile(ap,0.75,ind), quantile(ap,0.975,ind))
function Base.summary(ap::AsymptoticPosteriors.AsymptoticPosterior{N,T}; sigfigs=4) where {N,T}
    @static if VERSION < v"0.7-"
        sig(x, s) = signif(x, s)
    else
        sig(x, s) = round(x, sigdigits = s)
    end
    println("S1:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 1)) .+1) ./ 2, sigfigs))
    println("S2:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 2)) .+1) ./ 2, sigfigs))
    println("C1:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 3)) .+1) ./ 2, sigfigs))
    println("C2:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 4)) .+1) ./ 2, sigfigs))
    for i in 1:N-4
        println("Population $i proportion:\t", sig.(inv_logit.(four_num_sum(ap, i+4)), sigfigs))
    end
end
# function Base.summary(ap::AsymptoticPosteriors.AsymptoticPosterior{N,T,NoGoldDataCorr{N}}; sigfigs=4) where {N,T}
#     @static if VERSION < v"0.7-"
#         sig(x, s) = signif(x, s)
#     else
#         sig(x, s) = round(x, sigdigits = s)
#     end
#     println("S1:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 1)) .+1) ./ 2, sigfigs))
#     println("S2:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 2)) .+1) ./ 2, sigfigs))
#     println("C1:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 3)) .+1) ./ 2, sigfigs))
#     println("C2:\t\t\t\t", sig.((inv_logit.(four_num_sum(ap, 4)) .+1) ./ 2, sigfigs))
#     println("Cov1:\t\t\t\t", sig.(exp.(four_num_sum(ap, 5)), sigfigs))
#     println("Cov2:\t\t\t\t", sig.(exp.(four_num_sum(ap, 6)), sigfigs))
#     for i in 1:N-6
#         println("Population $i proportion:\t", sig.(inv_logit.(four_num_sum(ap, i+6)), sigfigs))
#     end
# end






# corr_errors_dependent2 = CorrErrors((0.1,        0.4), (0.9, 0.95), (0.85,0.97), (0.5,0.45));
# corr_errors_dependent3 = CorrErrors((0.1,  0.25, 0.4), (0.9, 0.95), (0.85,0.97), (0.5,0.45));
# corr_errors_dependent4 = CorrErrors((0.1,0.2,0.3,0.4), (0.9, 0.95), (0.85,0.97), (0.5,0.45));


# αS1 = 36
# βS1 = 4
# αS2 = 38
# βS2 = 2
# αC1 = 32
# βC1 = 8
# αC2 = 38
# βC2 = 2



# n_small_2c  = NoGoldDataCorr(corr_errors_dependent2,100,200,16,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);
# n_medium_2c = NoGoldDataCorr(corr_errors_dependent2,400,800,64,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);
# n_big_2c    = NoGoldDataCorr(corr_errors_dependent2,1600,3200,252,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);

# n_small_3c  = NoGoldDataCorr(corr_errors_dependent3,100,200,16,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);
# n_medium_3c = NoGoldDataCorr(corr_errors_dependent3,400,800,64,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);
# n_big_3c    = NoGoldDataCorr(corr_errors_dependent3,1600,3200,252,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);

# n_small_4c  = NoGoldDataCorr(corr_errors_dependent4,100,200,16,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);
# n_medium_4c = NoGoldDataCorr(corr_errors_dependent4,400,800,64,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);
# n_big_4c    = NoGoldDataCorr(corr_errors_dependent4,1600,3200,252,αS1=αS1,βS1=βS1,αS2=αS2,βS2=βS2,αC1=αC1,βC1=βC1,αC2=αC2,βC2=βC2);



