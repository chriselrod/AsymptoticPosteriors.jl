using AsymptoticPosteriors
using Base.Test

# write your own tests here
@test 1 == 2



using Revise, AsymptoticPosteriors
@code_warntype AsymptoticPosteriors.AsymptoticConfig(f2, max_density_params, Val{8}())
config = AsymptoticPosteriors.AsymptoticConfig(f2, max_density_params, Val{8}());

@code_warntype AsymptoticPosteriors.AsymptoticMAP(f2, max_density_params, config, Val{8}())
map = AsymptoticPosteriors.AsymptoticMAP(f2, max_density_params, config, Val{8}());


@code_warntype AsymptoticPosterior(map)
ap = AsymptoticPosterior(map)

ap = AsymptoticPosterior(f2, max_density_params, Val{8}())


function odgen(map)
    F = x -> begin
        AsymptoticPosteriors.arrays_equal(x, map.last_x) ? map.lastval[] : AsymptoticPosteriors.update_grad!(map, x)
    end

    G = (out, x) -> begin
        if  AsymptoticPosteriors.arrays_equal(x, map.last_x)
            grad = DiffResults.gradient(map.configuration.gresult)
            out === grad || copyto!(out, grad) #Does this make sense? Could the two objects end up being one and the same?
        else
            map.configuration.gresult.derivs = (out,)
            AsymptoticPosteriors.update_grad!(map, x)
        end
        out
    end

    FG = (out, x) -> begin
        if  AsymptoticPosteriors.arrays_equal(x, map.last_x)
            grad = DiffResults.gradient(map.configuration.gresult)
            out === grad || copyto!(out, grad) #Does this make sense? Could the two objects end up being one and the same?
        else
            map.configuration.gresult.derivs = (out,)
            AsymptoticPosteriors.update_grad!(map, x)
        end
        map.lastval[]
    end

    N = 8;

    T = Float64

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

@code_warntype odmax.f(initial_x)


# Why the union?!?!? Fix!
@code_warntype AsymptoticPosteriors.update_grad!(map, initial_x)

# Purterbation confusion!?!
AsymptoticPosteriors.update_grad!(map, initial_x)
# ERROR: Cannot determine ordering of Dual tags Void and ForwardDiff.Tag{#f2,Float64}
# Where does the Tag{#f2,Float64} dual tag come from?
# julia> typeof(map.configuration.theta)
# SubArray{ForwardDiff.Dual{ForwardDiff.Tag{#f2,Float64},Float64,8},1,Array{ForwardDiff.Dual{ForwardDiff.Tag{#f2,Float64},Float64,8},1},Tuple{UnitRange{Int64}},true}
