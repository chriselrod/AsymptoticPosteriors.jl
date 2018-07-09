
using Distributed
addprocs();
# addprocs(16);

@everywhere begin
# using Pkg
# include(joinpath(dir("AsymptoticPosteriors"), "examples/NGSCov.jl"));


include("/home/chris/.julia/dev/AsymptoticPosteriors/examples/NGSCov.jl")
using LineSearches, TriangularMatrices, Statistics
# using DifferentiableObjects

truth = CorrErrors((0.15, 0.4), (0.9, 0.95), (0.85,0.97));
data = NoGoldData(truth,1600,3200,252);
x = TriangularMatrices.randvec(6);
@time apc = AsymptoticPosterior(data, x, LineSearches.HagerZhang());
@time quantile(apc, 0.025)

# @time apc = AsymptoticPosterior(data, randn(6), Val(6), LineSearches.HagerZhang());
summary(apc)
end

@everywhere begin
function simulation_core(truth, k, iter, n_common = 100, n1 = 1000, n2 = 30)
    sim_results = Array{Float64}(undef,3,6,k)
    S₁, S₂, C₁, C₂, π₁, π₂ = truth.S[1], truth.S[2], truth.C[1], truth.C[2], truth.π[1], truth.π[2]

    init = RecursiveVector{Float64,6}()
    for i ∈ 1:4
        init[i] = truth.Θ[i]
    end
    for i ∈ 7:length(truth.Θ)
        init[i-2] = truth.Θ[i]
    end
    data = NoGoldData(truth, n_common, n1, n2)
    # resample!(data, truth, n_common, n1, n2)
    ap = AsymptoticPosterior(data, init, LineSearches.HagerZhang())
    
    
    AsymptoticPosteriors.fit!(ap.pl.map, init)
    set_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, 1)
    set_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, 1)
    set_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, 1)
    set_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, 1)
    set_buffer!(sim_results, inv_logit,       π₁, ap, 5, 1)
    set_buffer!(sim_results, inv_logit,       π₂, ap, 6, 1)
    @inbounds for j ∈ 2:iter
        resample!(data, truth, n_common, n1, n2)
        AsymptoticPosteriors.fit!(ap.pl.map, init)
        update_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, 1)
        update_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, 1)
        update_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, 1)
        update_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, 1)
        update_buffer!(sim_results, inv_logit,       π₁, ap, 5, 1)
        update_buffer!(sim_results, inv_logit,       π₂, ap, 6, 1)
    end
    for i ∈ 2:k
        n_common *= 2
        n1 *= 2
        n2 *= 2
        resample!(data, truth, n_common, n1, n2)
        AsymptoticPosteriors.fit!(ap.pl.map, init)
        set_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, i)
        set_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, i)
        set_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, i)
        set_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, i)
        set_buffer!(sim_results, inv_logit,       π₁, ap, 5, i)
        set_buffer!(sim_results, inv_logit,       π₂, ap, 6, i)
        @inbounds for j ∈ 2:iter
            resample!(data, truth, n_common, n1, n2)
            AsymptoticPosteriors.fit!(ap.pl.map, init)
            update_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, i)
            update_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, i)
            update_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, i)
            update_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, i)
            update_buffer!(sim_results, inv_logit,       π₁, ap, 5, i)
            update_buffer!(sim_results, inv_logit,       π₂, ap, 6, i)
        end
    end
    sim_results
end
function simulation_core(truth, K1,K2,K3, iter, n_common_base = 100, n1_base = 1000, n2_base = 30)
    sim_results = Array{Float64}(undef,3,6,k)
    S₁, S₂, C₁, C₂, π₁, π₂ = truth.S[1], truth.S[2], truth.C[1], truth.C[2], truth.π[1], truth.π[2]

    init = RecursiveVector{Float64,6}()
    for i ∈ 1:4 # Set truth as initial guess.
        init[i] = truth.Θ[i]
    end
    for i ∈ 7:length(truth.Θ)
        init[i-2] = truth.Θ[i]
    end
    data = NoGoldData(truth)
    # resample!(data, truth, n_common, n1, n2)
    ap = AsymptoticPosterior(undef, data, init, LineSearches.HagerZhang()) #not yet fit on data
    
    for k3 ∈ 1:K3, k2 ∈ 1:K2, k1 ∈ 1:K1
        n_common = round(Int, n_common_base * √2^(k1-1))
        n1 = round(Int, n1_base * √2^(k2-1))
        n2 = round(Int, n2_base * √2^(k3-1))
        resample!(data, truth, n_common, n1, n2)
        AsymptoticPosteriors.fit!(ap.pl.map, init)
        set_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, k1,k2,k3)
        set_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, k1,k2,k3)
        set_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, k1,k2,k3)
        set_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, k1,k2,k3)
        set_buffer!(sim_results, inv_logit,       π₁, ap, 5, k1,k2,k3)
        set_buffer!(sim_results, inv_logit,       π₂, ap, 6, k1,k2,k3)
        @inbounds for j ∈ 2:iter
            resample!(data, truth, n_common, n1, n2)
            AsymptoticPosteriors.fit!(ap.pl.map, init)
            update_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, k1,k2,k3)
            update_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, k1,k2,k3)
            update_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, k1,k2,k3)
            update_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, k1,k2,k3)
            update_buffer!(sim_results, inv_logit,       π₁, ap, 5, k1,k2,k3)
            update_buffer!(sim_results, inv_logit,       π₂, ap, 6, k1,k2,k3)
        end
    end
    sim_results
end
function simulation_core(truth, k, iter, n_common = 100, n1 = 1000, n2 = 30)
    sim_results = Array{Float64}(undef,3,6,k)
    S₁, S₂, C₁, C₂, π₁, π₂ = truth.S[1], truth.S[2], truth.C[1], truth.C[2], truth.π[1], truth.π[2]

    init = RecursiveVector{Float64,6}()
    for i ∈ 1:4
        init[i] = truth.Θ[i]
    end
    for i ∈ 7:length(truth.Θ)
        init[i-2] = truth.Θ[i]
    end
    data = NoGoldData(truth, n_common, n1, n2)
    # resample!(data, truth, n_common, n1, n2)
    ap = AsymptoticPosterior(data, init, LineSearches.HagerZhang())
    
    
    AsymptoticPosteriors.fit!(ap.pl.map, init)
    set_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, 1)
    set_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, 1)
    set_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, 1)
    set_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, 1)
    set_buffer!(sim_results, inv_logit,       π₁, ap, 5, 1)
    set_buffer!(sim_results, inv_logit,       π₂, ap, 6, 1)
    @inbounds for j ∈ 2:iter
        resample!(data, truth, n_common, n1, n2)
        AsymptoticPosteriors.fit!(ap.pl.map, init)
        update_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, 1)
        update_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, 1)
        update_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, 1)
        update_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, 1)
        update_buffer!(sim_results, inv_logit,       π₁, ap, 5, 1)
        update_buffer!(sim_results, inv_logit,       π₂, ap, 6, 1)
    end
    for i ∈ 2:k
        n_common *= 2
        n1 *= 2
        n2 *= 2
        resample!(data, truth, n_common, n1, n2)
        AsymptoticPosteriors.fit!(ap.pl.map, init)
        set_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, i)
        set_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, i)
        set_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, i)
        set_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, i)
        set_buffer!(sim_results, inv_logit,       π₁, ap, 5, i)
        set_buffer!(sim_results, inv_logit,       π₂, ap, 6, i)
        @inbounds for j ∈ 2:iter
            resample!(data, truth, n_common, n1, n2)
            AsymptoticPosteriors.fit!(ap.pl.map, init)
            update_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, i)
            update_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, i)
            update_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, i)
            update_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, i)
            update_buffer!(sim_results, inv_logit,       π₁, ap, 5, i)
            update_buffer!(sim_results, inv_logit,       π₂, ap, 6, i)
        end
    end
    sim_results
end

function update_buffer!(buffer::AbstractArray{T}, f, true_val, ap, j, k...) where T
    l = f(quantile(ap, 0.025, j))
    u = f(quantile(ap, 0.975, j))
    if (l > true_val) | (u < true_val)
        buffer[1,j,k...] += one(T)
    end
    buffer[2,j,k...] += u - l
    buffer[3,j,k...] += abs(true_val- f(ap.pl.map.θhat[j]))
end
function set_buffer!(buffer::AbstractArray{T}, f, true_val, ap, j, k...) where T
    l = f(quantile(ap, 0.025, j))
    u = f(quantile(ap, 0.975, j))
    buffer[1,j,k...] = ifelse((l > true_val) | (u < true_val), one(T), zero(T)) # need to set it to zero.
    buffer[2,j,k...] = u - l
    buffer[3,j,k...] = abs(true_val- f(ap.pl.map.θhat[j]))
end

@inline inv_uhalf_logit(x) = 0.5+0.5/(1+exp(-x ))


end #everywhere

function run_simulation(truth, k = 5, iter = 25, n_common = 100, n1 = 1000, n2 = 30)
    @distributed (+) for i ∈ 2:nprocs()
        simulation_core(truth, k, iter, n_common, n1, n2)
    end
end
function run_simulation2(truth, k = 5, iter = 50, n_common = 100, n1 = 1000, n2 = 30)
    @distributed (+) for i ∈ 2:nprocs()
        simulation_core(truth, k,k,k, iter, n_common, n1, n2)
    end
end
