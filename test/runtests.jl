using AsymptoticPosteriors
using Base.Test

# write your own tests here
@test 1 == 2



using AsymptoticPosteriors
include("/home/chris/Documents/progwork/julia/NGS.jl")
f3(x) = NGS(cdb4, x, Val{8}())
@time ap = AsymptoticPosterior(f3, initial_x, Val(8));
quantile(ap, 0.025)


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


