using AsymptoticPosteriors
using Base.Test

# write your own tests here
@test 1 == 2




include("/home/chris/Documents/progwork/julia/NGSCov.jl");
include("/home/chris/Documents/progwork/julia/NGStan.jl");
@time ap = AsymptoticPosterior(n_big_4, init8, Val(8));
@time apc = AsymptoticPosterior(n_big_4c, initCorr10, Val(10));
summary(ap)
summary(apc)

using BenchmarkTools
# @benchmark AsymptoticPosterior($n_big_4, $init8, Val(8))
# @benchmark four_num_sum.(($ap,), 1:8)
@benchmark AsymptoticPosterior($n_big_2c, $initCorr8, Val(8))
@benchmark four_num_sum.(($apc,), 1:10)


low_sensitivity = CorrErrors((0.1,0.2,0.3,0.4), (0.8, 0.8), (0.98,0.98));
lowS1 = NoGoldDataCorr(low_sensitivity,100,200,16);
ap_lows1 = AsymptoticPosterior(lowS1, initCorr10, Val(10));
summary(ap_lows1)
rstan(lowS1, 10_000)
using RCall
R"options(width=160)"
summary(ap_lows1)
R"summary(res)$summary"

low_specificity = CorrErrors((0.1,0.2,0.3,0.4), (0.98, 0.98), (0.8,0.8));
lowC1 = NoGoldDataCorr(low_specificity,100,200,16);
ap_lowc1 = AsymptoticPosterior(lowC1, initCorr10, Val(10));
summary(ap_lowc1)
rstan(lowC1, 10_000)
summary(ap_lowc1)
R"summary(res)$summary"

n =  [13   12; 17   20; 11   11; 59   57; 279  308; 721  692; 11   15; 19   15];
test_data = NoGoldDataCorr{8}(n =n,
αS1 = 16.0, βS1 = 4.0,
αS2 = 16.0, βS2 = 4.0,
αC1 = 16.0, βC1 = 4.0,
αC2 = 16.0, βC2 = 4.0);
@time apc = AsymptoticPosterior(test_data, initCorr8, Val(8));
summary(apc)

using AsymptoticPosteriors
include("/home/chris/Documents/progwork/julia/NGS.jl")
f3(x) = NGS(cdb4, x, Val{8}())
@time ap = AsymptoticPosterior(f3, initial_x, Val(8));
quantile(ap, 0.025)

interval(ap, ind) = (quantile(ap, 0.025, ind), quantile(ap, 0.975, ind))
@benchmark interval($ap, 1)
@benchmark interval($ap, 2)
@benchmark interval($ap, 3)
@benchmark interval($ap, 4)
@benchmark interval($ap, 5)
@benchmark interval($ap, 6)
@benchmark interval($ap, 7)
@benchmark interval($ap, 8)

@benchmark AsymptoticPosterior(f3, $initial_x, Val(8))
@benchmark AsymptoticPosteriors.fit!($ap.pl.map, $initial_x)


using Revise, AsymptoticPosteriors
using Traceur
@code_warntype AsymptoticPosteriors.AsymptoticConfig(f2, max_density_params, Val{8}())
config = @trace AsymptoticPosteriors.AsymptoticConfig(f2, max_density_params, Val{8}());
# config = AsymptoticPosteriors.AsymptoticConfig(f2, max_density_params, Val{8}());

@code_warntype AsymptoticPosteriors.AsymptoticMAP(max_density_params, config, Val{8}())
# map = @trace AsymptoticPosteriors.AsymptoticMAP(max_density_params, config, Val{8}());
map = AsymptoticPosteriors.AsymptoticMAP(max_density_params, config, Val{8}());


@code_warntype AsymptoticPosterior(map)
# ap = AsymptoticPosterior(map)

# ap = AsymptoticPosterior(f2, max_density_params, Val{8}())


function odgen(map::M) where {T, N, M <: AsymptoticPosteriors.AsymptoticMAP{N,T}}
    F = x -> begin
        map::M
        AsymptoticPosteriors.arrays_equal(x, map.last_x) ? map.lastval[] : AsymptoticPosteriors.update_grad!(map, x)
    end

    G = (out, x) -> begin
        AsymptoticPosteriors.update_grad!(out, map::M, x)
        out
    end

    FG = (out, x) -> begin
        map::M
        AsymptoticPosteriors.update_grad!(out, map, x)
        map.lastval[]
    end

    initial_x = Vector{T}(undef, N+1)

    @inbounds begin
        for i ∈ 1:N
            initial_x[i] = map.θhat[i]
        end
        initial_x[end] = zero(T)
    end
    OnceDifferentiable(F, G, FG, initial_x, zero(T), Val{true}())
end

odmax = odgen(map);

backtrack = BFGS(LineSearches.InitialStatic(), LineSearches.BackTracking(), x -> Matrix{eltype(x)}(I, length(x), length(x)), Optim.Flat())

options = Optim.InternalUseOptions(backtrack)

# state = Optim.initial_state(backtrack, options, odmax, initial_x)

fieldnames(typeof(odmax))

xtest = vcat(max_density_params, [1]);
theta = @view(xtest[1:8]);
out = similar(xtest);

AsymptoticPosteriors.update_hessian!(map, theta)
AsymptoticPosteriors.qb(map, theta )

@code_warntype odmax.f(xtest)

@code_warntype AsymptoticPosteriors.update_grad!(map, xtest)

# Purterbation confusion!?!
AsymptoticPosteriors.update_grad!(map, max_density_params)
# ERROR: Cannot determine ordering of Dual tags Void and ForwardDiff.Tag{#f2,Float64}
# Where does the Tag{#f2,Float64} dual tag come from?
# julia> typeof(map.configuration.theta)
# SubArray{ForwardDiff.Dual{ForwardDiff.Tag{#f2,Float64},Float64,8},1,Array{ForwardDiff.Dual{ForwardDiff.Tag{#f2,Float64},Float64,8},1},Tuple{UnitRange{Int64}},true}

# Run through WTF theta is supposed to be in each use case! Seems wrong.




ap = AsymptoticPosterior(f2, max_density_params, Val{8}());

ap.map.θhat |> detransform_res


AsymptoticPosteriors.bound(ap, 1, 0.025)













precise_options =  Optim.Options(g_tol = 1e-30,x_tol=0.0,f_tol=0.0)
opt_precise = optimize(f2, BigFloat.(max_density_params), hage, precise_options)
prex_max_dens = Optim.minimizer(opt_precise)

xbf = prex_max_dens[1]
mdp_bf = prex_max_dens[2:end]


opt_1 = optimize(y -> f2(vcat(xbf-eps()/2,y)), mdp_bf, hage, precise_options)
min1 = Optim.minimizer(opt_1)
minimum_1 = Optim.minimum(opt_1)
opt_2 = optimize(y -> f2(vcat(xbf+eps()/2,y)), mdp_bf, hage, precise_options)
min2 = Optim.minimizer(opt_2)
minimum_2 = Optim.minimum(opt_2)

grad = @. (min2 - min1)/eps()
dlp = (minimum_2 - minimum_1) / eps()

precise_hessian = ForwardDiff.hessian(f2, Float64.(prex_max_dens))
estimated_grad = -inv(BigFloat.(precise_hessian[2:end,2:end]))*BigFloat.(precise_hessian[2:end,1])
grad .- estimated_grad

precise_grad = ForwardDiff.gradient(f2, Float64.(prex_max_dens))
estimated_dlp = precise_grad' * vcat([1.0],estimated_grad)
dlp - estimated_dlp


xbf2 = xbf - 1.96 * inv(precise_hessian)[1]
opt_precise2 = optimize(y -> f2(vcat(xbf2,y)), mdp_bf, hage, precise_options)
prex_max_dens2 = Optim.minimizer(opt_precise2)
prex_max_dens2_ = vcat(xbf2, prex_max_dens2)

opt_3 = optimize(y -> f2(vcat(xbf2-eps()/2,y)), mdp_bf, hage, precise_options)
min3 = Optim.minimizer(opt_3)
minimum_3 = Optim.minimum(opt_3)
opt_4 = optimize(y -> f2(vcat(xbf2+eps()/2,y)), mdp_bf, hage, precise_options)
min4 = Optim.minimizer(opt_4)
minimum_4 = Optim.minimum(opt_4)

grad2 = @. (min4 - min3)/eps()
dlp2 = (minimum_4 - minimum_3) / eps()

precise_hessian2 = ForwardDiff.hessian(f2, Float64.(prex_max_dens2_))
estimated_grad2 = -inv(BigFloat.(precise_hessian2[2:end,2:end]))*BigFloat.(precise_hessian2[2:end,1])
grad2 .- estimated_grad2

precise_grad2 = ForwardDiff.gradient(f2, Float64.(prex_max_dens2_))
estimated_dlp2 = precise_grad2' * vcat([1.0],estimated_grad2)
dlp2 - estimated_dlp2


phi = inv(precise_hessian2);
estimated_dlp3 = precise_grad2[1] / (phi[1] * precise_hessian2[1] )


