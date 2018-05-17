# AsymptoticPosteriors

[![Build Status](https://travis-ci.org/chriselrod/AsymptoticPosteriors.jl.svg?branch=master)](https://travis-ci.org/chriselrod/AsymptoticPosteriors.jl)

[![Coverage Status](https://coveralls.io/repos/chriselrod/AsymptoticPosteriors.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chriselrod/AsymptoticPosteriors.jl?branch=master)

[![codecov.io](http://codecov.io/github/chriselrod/AsymptoticPosteriors.jl/coverage.svg?branch=master)](http://codecov.io/github/chriselrod/AsymptoticPosteriors.jl?branch=master)


---


This library implements the third order asymptotic approximation to a posterior distribution from [Approximate Bayesian computation with modified log-likelihood ratios](http://www.utstat.utoronto.ca/reid/research/Metron-published.pdf) by Laura Ventura and Nancy Reid, with the caveat that I did not draw a distinction between the priors and likelihood, lumping the two together (under the likelihood).
This has two chief advantages. Firstly, in the case of poorly identified likelihoods, prior information can help stabilize the optimization for the repeated maximizations and ensure positive definite Hessian matrices (otherwise, the rootfinding and optimization process may step into a part of the posterior where it is computationally singular). Secondly, it allows the user to write a single "log posterior" function, and provide any necessary transformations. This function must take a parameter vector as its only input.

It is strongly recomended to transform the parameters to an unconstrained parameter space. I would suggest through [ContinuousTransformations](https://github.com/tpapp/ContinuousTransformations.jl). This ensures that the root finding and optimization algorithms do not step outside of the parameter space, and is also likely to increase the accuracy of asymptotic approximations by brining the distribution closer to normality.


Compile times are very bad on Julia 0.7. They're also 10x worse on Julia 0.6. I'll work on refactoring code so that we at least do not have to recompile the same code for different likelihood functions. I encouraged individual specialization aggressively, because my focus was on simulations, and I did not anticipate how slow compilation could be when this was taken to the extreme.

That said, I do support Julia 0.7, and on top of far more pallatable compile times, it runs much more quickly too. I imagine there was an optimization added making it smarter about deciding when not to specialize.


This package has a `AutoDiffDifferentiable` object that is uses to wrap the likelihood function, and uses [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) for automatic differentiation. I will definitely add [Capstan](https://github.com/JuliaDiff/Capstan.jl) support as soon as it is released, and am considering supporting [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl) for reverse mode AD, and [Reduce](https://github.com/chakravala/Reduce.jl) for symbolic differentiation. Capstan and Reduce both have a lot of promise. In Reduce's case, because of our heavy dependency on repeatedly evaluating Hessians which is rather inefficient with ForwardDiff.

This `AutoDiffDifferentiable` object can work in place of [NLSolversBase](https://github.com/JuliaNLSolvers/NLSolversBase.jl)'s differentiable objects for [optimization](https://github.com/JuliaNLSolvers/Optim.jl). BFGS with third order backtracking is used for finding the global maximum, while BFGS with HagerZhang's [line search](https://github.com/JuliaNLSolvers/LineSearches.jl) is used for the profile maximization, for greater stability. False Position is the root finding algorithm for finding credible intervals.

`AutoDiffDifferentiable` objects are also callable. They return an immutable diff result with a copy of the gradient. This allows support for [DynamicHMC](https://github.com/tpapp/DynamicHMC.jl). The asymptotic posteriors objects behave the same way.

I am likely to move these objects to a separate backage if I add much more functionality.

---

Lets look at a brief example. The example scripts depend on [Parameters](https://github.com/mauro3/Parameters.jl) and [Rmath](https://github.com/JuliaStats/Rmath.jl) in addition to AsymptoticPosterior's requirements. You may already have these installed, but:

```julia
Pkg.update()
Pkg.clone("AsymptoticPosteriors")
Pkg.add("Parameters")
Pkg.add("Rmath")
```

The example uses the following misclassification model:
```julia
function NoGoldStandard(priordata, S₁::T, S₂::T, C₁::T, C₂::T, π::Vararg{T,K}) where {T,K}
    @unpack n, απ, βπ, αS1, βS1, αS2, βS2,
        αC1, βC1, αC2, βC2 = priordata
    target = beta_lpdf(S₁, αS1, βS1) + beta_lpdf(S₂, αS2, βS2)
    target += beta_lpdf(C₁, αC1, βC1) + beta_lpdf(C₂, αC2, βC2)
    # target += sum(pdf.(Beta( απ, βπ), pi))
    # target += pdf(Beta( αS1, βS1), S₁)
    # target += pdf(Beta( αS2, βS2), S₂)
    # target += pdf(Beta( αC1, βC1), C₁)
    # target += pdf(Beta( αC2, βC2), C₂)
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
```
We have K populations, each with a different prevalence `πₖ` on a state of interest. We have two tests with sensitivities, S, and specificities, C. We have samples that underwent both tests, as well as samples that underwent only one of the two.

We can run the model and the boiler plate (ie, unconstraining transformations and Jacobians) via
```julia
include(joinpath(Pkg.dir("AsymptoticPosteriors"),"examples","NGSCov.jl"))
```
this also loads a few convenience objects for data generation:

```julia
truth = CorrErrors((0.1, 0.4), (0.9, 0.95), (0.85,0.97));
data = NoGoldData(truth,1600,3200,252);
```
This species the truth as two populations, with prevalences of 0.1 and 0.4, and two tests which have sensitivities of 0.9 and 0.95 respectively, and specifities of 0.85 and 0.97. 
We then generate a dataset with 160 observations from both populations, 320 from only the first test, and 25 from only the second.

Then, to fit the model:
```
ap = AsymptoticPosterior(data, fill(0.0,6), Val(6));
summary(ap)
```

We can compare these results to MCMC:
```julia
using DynamicHMC, MCMCDiagnostics
sample, NUTS_tuned = NUTS_init_tune_mcmc(ap, mode(ap), 1000);
```
This model is not well behaved, and I get errors without good starting values. Therefore I seed the chain with the mode. Even then, you may have to rerun this several times to avoid errors.
Rearranging to a more convenient form:
```julia
posterior = SVector{6}.(inverse_transform.(get_position.(sample)));
posterior_matrix = reshape(reinterpret(Float64, posterior), (6,1000));
```
we can see that the results are rather comparable:
```julia
julia> effective_sample_size(first.(posterior))
726.4754846982462

julia> for i ∈ 1:6
           println(i, "\t", signif.(quantile.((posterior_matrix[i,:],), (0.025,0.25,0.75,0.975)), (4,)))
       end
1	(0.8938, 0.9203, 0.9472, 0.9728)
2	(0.8758, 0.9179, 0.9601, 0.9916)
3	(0.8408, 0.851, 0.8636, 0.8749)
4	(0.9506, 0.9608, 0.9703, 0.9794)
5	(0.08215, 0.09289, 0.1063, 0.1195)
6	(0.3465, 0.3634, 0.3835, 0.4032)

julia> summary(ap)
S1:				(0.8954, 0.9206, 0.9469, 0.9722)
S2:				(0.8773, 0.9169, 0.9581, 0.9912)
C1:				(0.8403, 0.8513, 0.8638, 0.8763)
C2:				(0.9504, 0.9604, 0.9704, 0.9792)
Population 1 proportion:	(0.08219, 0.09338, 0.1065, 0.1203)
Population 2 proportion:	(0.3458, 0.3641, 0.3841, 0.4036)
```
It will take a little work to get MCMC working consistently for smaller sample sizes. Stan tends to work without issue, and results are comparable at small sample sizes as well. Increasing the number of MCMC samples tends to make results closer.

---

Why consider this over MCMC? The primary advantage is speed. The example `simulation.jl` will run a simulation showing us how 95% credible interval error rates, 95% interval width, and bias of point estimates change as a function of sample size. Note, including this runs `addprocs()` and compiles the asymptotic posterior and summary function on all available cores.

The arguments of the function `run_simulation` are 1) a set of true values, 2) a number of times to double the sample size, 3) number of iterations per core, 4-6) initial sample sizes in common, for population 1 only, and for population 2 only.
This script currently requires Julia 0.7.
```julia
using Pkg
include(joinpath(dir("AsymptoticPosteriors"),"examples","simulation.jl"))
@time sim_results = run_simulation(truth, 6, 25, 50, 500, 15)
```