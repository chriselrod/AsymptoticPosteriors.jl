using OhMyREPL
using Distributed
# addprocs(8);
addprocs();
# addprocs(16);


@everywhere begin
# using LineSearches, TriangularMatrices, Statistics
using Statistics, SIMDArrays
using AsymptoticPosteriors, Parameters, Rmath
using Base.Cartesian, Random, LinearAlgebra
# using Pkg
# include(joinpath(dir("AsymptoticPosteriors"), "examples/NGSCov.jl"));
end
@everywhere begin
# include("/home/chris/.julia/dev/AsymptoticPosteriors/examples/NGSCov.jl")
include("/home/chriselrod/.julia/dev/AsymptoticPosteriors/examples/NGSCov.jl")
# using DifferentiableObjects

const truth = CorrErrors((0.15, 0.4), (0.9, 0.95), (0.85,0.97));
const data = NoGoldData(truth,160,320,52);
x = randnsimd(6);
end
@everywhere begin
@time const ap = AsymptoticPosterior(data, x);
end
@everywhere begin
@time quantile(ap, 0.025)

# @time apc = AsymptoticPosterior(data, randn(6), Val(6), LineSearches.HagerZhang());
summary(ap)
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
    sim_results = Array{Float64}(undef,3,6,K1,K2,K3)
    S₁, S₂, C₁, C₂, π₁, π₂ = truth.S[1], truth.S[2], truth.C[1], truth.C[2], truth.π[1], truth.π[2]

    init = RecursiveVector{Float64,6}()
    for i ∈ 1:4 # Set truth as initial guess.
        init[i] = truth.Θ[i]
    end
    for i ∈ 7:length(truth.Θ)
        init[i-2] = truth.Θ[i]
    end
    # data = NoGoldData{6}(
    #     n = Matrix{Int}(undef,8,$2),
    #     απ = 2.0, # απ
    #     βπ = 2.0, # βπ
    #     αS1 = 2.0, # αS1
    #     βS1 = 2.0, # βS1
    #     αS2 = 2.0, # αS2
    #     βS2 = 2.0, # βS2
    #     αC1 = 2.0, # αC1
    #     βC1 = 2.0, # βC1
    #     αC2 = 2.0, # αC2
    #     βC2 = 2.0, # βC2
    # )
    data = NoGoldData(truth)
    # resample!(data, truth, n_common, n1, n2)
    ap = AsymptoticPosterior(undef, data, init, LineSearches.HagerZhang()) #not yet fit on data

    for k3 ∈ 1:K3, k2 ∈ 1:K2, k1 ∈ 1:K1
        n_common = round(Int, n_common_base * 2^((k1-1)//8))
        n1 = round(Int, n1_base * 2^((k2-1)//8))
        n2 = round(Int, n2_base * 2^((k3-1)//8))
        resample!(data, truth, n_common, n1, n2)
        # try
            AsymptoticPosteriors.fit!(ap.pl.map, init)
            set_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, k1,k2,k3)
            set_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, k1,k2,k3)
            set_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, k1,k2,k3)
            set_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, k1,k2,k3)
            set_buffer!(sim_results, inv_logit,       π₁, ap, 5, k1,k2,k3)
            set_buffer!(sim_results, inv_logit,       π₂, ap, 6, k1,k2,k3)
        # catch err
        #     @show data
        #     @show n_common, n1, n2
        #     rethrow(err)
        # end
        @inbounds for j ∈ 2:iter
            resample!(data, truth, n_common, n1, n2)
            # try
                AsymptoticPosteriors.fit!(ap.pl.map, init)
                update_buffer!(sim_results, inv_uhalf_logit, S₁, ap, 1, k1,k2,k3)
                update_buffer!(sim_results, inv_uhalf_logit, S₂, ap, 2, k1,k2,k3)
                update_buffer!(sim_results, inv_uhalf_logit, C₁, ap, 3, k1,k2,k3)
                update_buffer!(sim_results, inv_uhalf_logit, C₂, ap, 4, k1,k2,k3)
                update_buffer!(sim_results, inv_logit,       π₁, ap, 5, k1,k2,k3)
                update_buffer!(sim_results, inv_logit,       π₂, ap, 6, k1,k2,k3)
            # catch err
            #     @show data
            #     @show n_common, n1, n2
            #     rethrow(err)
            # end
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

const increments = (
    ((4,0),  (-4,0), (0, 0) ),
    ((4,0),  (0,0),  (-4, 0)),
    ((-4,0), (8,0),  (0, 0) ),
    ((-4,0), (0,0),  (8, 0) ),
    ((0,0),  (4,0),  (0, 0) ),
    ((0,0),  (0,0),  (10, 0)),
    ((2,0),  (0,0),  (0, 0) ),
    ((0,4),  (0,-4), (0, 0) ),
    ((0,4),  (0,0),  (0,-4) ),
    ((0,-4), (0,8),  (0, 0) ),
    ((0,-4), (0,0),  (0,8)  ),
    ((0,0),  (0,4),  (0, 0) ),
    ((0,0),  (0,0),  (0, 4) ),
    ((0,2),  (0,0),  (0, 0) )
)

function increment_n(n_common, n1, n2, i)
    increment = increments[i]
    nctemp = n_common .+ increment[1]
    n1temp = n1 .+ increment[2]
    n2temp = n2 .+ increment[3]
    nctemp, n1temp, n2temp
end
function cost((n1cost,n2cost), n_common, n1, n2)
    n1cost * sum(n_common .+ n1) + n2cost * sum(n_common .+ n2)
end
function increment_cost((n1cost,n2cost), i)
    n_common, n1, n2 = increments[i]
    n1cost * sum(n_common .+ n1) + n2cost * sum(n_common .+ n2)
end

function average_loss(costs, iter, n_common, n1, n2)
    sim_results = Array{Float64}(undef,length(increments))
    S₁, S₂, C₁, C₂, π₁, π₂ = truth.S[1], truth.S[2], truth.C[1], truth.C[2], truth.π[1], truth.π[2]

    init = RecursiveVector{Float64,6}()
    for i ∈ 1:4
        init[i] = truth.Θ[i]
    end
    for i ∈ 7:length(truth.Θ)
        init[i-2] = truth.Θ[i]
    end
    # uses global data and ap
    for i ∈ 1:length(increments)
        nctemp, n1temp, n2temp = increment_n(n_common, n1, n2, i)
        if any(nctemp .< 0) || any(n1temp .< 0) || any(n2temp .< 0)
            sim_results[i] = Inf
            continue
        end
        sim_results[i] = increment_cost(costs, i)*iter
        @inbounds for k ∈ 1:iter
            resample!(data, truth, nctemp, n1temp, n2temp)
            AsymptoticPosteriors.fit!(ap.pl.map, init)
            sim_results[i] += loss(ap, init)
        end
    end
    sim_results# ./ iter
end

# @inline lossv2(v, truth, p) = v > truth ? p*(v-truth) : (1-p)*(truth-v)
@inline loss(v, truth, p) = (p - (v < truth))*(v-truth)
loss((l, u)::Tuple{<:Number,<:Number}, truth, pl = 0.025, pu = 0.975) = loss(l, truth, pl) + loss(u, truth, pu)

function loss(ap::AsymptoticPosteriors.AsymptoticPosterior, truth)
    l = 0.0
    @inbounds for i ∈ eachindex(truth)
        l += loss((quantile(ap, 0.025, i),quantile(ap, 0.975, i)), truth[i])
    end
    l
end

end #everywhere

function run_simulation(truth, k = 5, iter = 25, n_common = 100, n1 = 1000, n2 = 30)
    @distributed (+) for i ∈ 2:nprocs()
        simulation_core(truth, k, iter, n_common, n1, n2)
    end
end
function run_simulation2(truth, k = 10, iter = 50, n_common = 100, n1 = 500, n2 = 30)
    @distributed (+) for i ∈ 2:nprocs()
        simulation_core(truth, k,k,k, iter, n_common, n1, n2)
    end
end


# @time @everywhere begin
# const data = NoGoldData(truth)
# const ap = AsymptoticPosterior(undef, data, x, LineSearches.HagerZhang())
# end #everywhere

function simulation_search( costs, num_iter, n_common = (100,100), n1 = (200,200), n2 = (30,30) )
    nproc = nprocs()
    current_loss = Inf
    losses = @distributed (+) for i ∈ 2:nproc
        average_loss(costs, num_iter, n_common, n1, n2)
    end
    last_loss, minind = findmin(losses)
    iteration = 1
    while last_loss < current_loss
        @show iteration, losses
        @show n_common, n1, n2
        iteration += 1
        current_loss = last_loss
        n_common, n1, n2 = increment_n(n_common, n1, n2, minind)
        losses = @distributed (+) for i ∈ 2:nproc
            average_loss(costs, num_iter, n_common, n1, n2)
        end
        last_loss, minind = findmin(losses)
        if last_loss > current_loss # We'll try again.
            println("Failed to find best argument. Searching broadly.")
            for n ∈ 1:length(increments)
                n_common_temp, n1_temp, n2_temp = increment_n(n_common, n1, n2, n)
                (any(n_common_temp .< 0) || any(n1_temp .< 0) || any(n2_temp .< 0)) && continue
                losses_temp = @distributed (+) for i ∈ 2:nproc
                    average_loss(costs, num_iter, n_common_temp, n1_temp, n2_temp)
                end
                @show n_common_temp, n1_temp, n2_temp
                @show losses_temp
                last_loss_temp, minind_temp = findmin(losses_temp)
                last_loss_temp += increment_cost(costs, n) * num_iter * (nproc-1)
                if last_loss_temp < current_loss
                    last_loss = last_loss_temp
                    n_common, n1, n2 = n_common_temp, n1_temp, n2_temp
                    minind = minind_temp
                    break
                end
            end

        end
    end
    @show iteration, losses
    n_common, n1, n2, current_loss
end


# @time simulation_search((0.0000005,0.005), 1000, (150,150), (5,5), (5,5))
# 4869.788621 seconds (380.42 k allocations: 35.323 MiB, 0.00% gc time)
# ((350, 320), (75, 15), (5, 5), 441712.07542121154)

# 138 137
# 274 235
# 898 275
# 942 283
# 964* 900
# 968* 944
# 971* 966*
# 973* 970*
# 1012 973*
# 1014* 975*
# 1159 1014
# 1295 1016*
# 1335 1161
#      1297
#      1337
#
