
abstract type ForwardDiffDifferentiable <: NLSolversBase.AbstractObjective end


struct Configuration{N,T,T2,V,ND,DJ,DG,DG,F}
    f::F
    result::DiffResults.MutableDiffResult{V,Tuple{Vector{V},Matrix{V}}}
    inner_result::DiffResults.MutableDiffResult{Dual{T,V,ND},Tuple{Vector{Dual{T,V,ND}}}}
    jacobian_config::ForwardDiff.JacobianConfig{T,V,ND,DJ}
    gradient_config::ForwardDiff.GradientConfig{T,Dual{T,V,ND},ND,DG}
    gconfig::ForwardDiff.GradientConfig{T2,V,ND,DG}
end

struct LeanDifferentiable{TF, TDF, TX, Tcplx<:Union{Val{true},Val{false}},C} <: ForwardDiffDifferentiable
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    x_h::TX #??
    config::C
    # f_calls::Vector{Int}
    # df_calls::Vector{Int}
end

struct ProfileDifferentiable{TF, TDF, TX, Tcplx<:Union{Val{true},Val{false}},C} <: ForwardDiffDifferentiable
    x_f::TX # x used to evaluate f (stored in F)
    x_df::TX # x used to evaluate df (stored in DF)
    x_h::TX #??
    config::C
    # f_calls::Vector{Int}
    # df_calls::Vector{Int}
end


DiffResults.value!(obj::ForwardDiffDifferentiable, x::Real) = DiffResults.value!(obj.config.result, x)

NLSolversBase.value(obj::ForwardDiffDifferentiable) = DiffResults.value(obj.config.result)
NLSolversBase.gradient(obj::ForwardDiffDifferentiable) = DiffResults.gradient(obj.config.result)
NLSolversBase.gradient(obj::ForwardDiffDifferentiable, i::Integer) = DiffResults.gradient(obj.config.result)[i]
NLSolversBase.hessian(obj::ForwardDiffDifferentiable) = DiffResults.hessian(obj.config.result)

function f(obj, x)
    obj.config.f(x)
end
function df(obj::ForwardDiffDifferentiable, x)
    ForwardDiff.gradient!(gradient(obj), obj.config.f, x, obj.config.gconfig, Val{false}())
end
function fdf(obj::ForwardDiffDifferentiable, x)
    obj.config.result.derivs = (gradient(obj), hessian(obj))
    ForwardDiff.gradient!(obj.config.result, obj.config.f, x, obj.config.gconfig, Val{false}())
    DiffResults.value(obj.config.result)
end

function Configuration(f::F, x::AbstractArray{V}, chunk::Chunk = Chunk(x), tag = Tag(f, V)) where {F,V}
    result = DiffResults.HessianResult(x)
    jacobian_config = ForwardDiff.JacobianConfig((f,gradient), DiffResults.gradient(result), x, chunk, tag)
    gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    inner_result = DiffResults.DiffResult(zero(eltype(jacobian_config.duals[2])), jacobian_config.duals[2])
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)
    Configuration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
end

function (c::Configuration)(y, z)
    c.inner_result.derivs = (y,) #Already true?
    ForwardDiff.gradient!(c.inner_result, c.f, z, c.gradient_config, Val{false}())
    DiffResults.value!(c.result, ForwardDiff.value(DiffResults.value(c.inner_result)))
    return y
end
function hessian!(c::Configuration, x::AbstractArray) where {T,CHK}
    ForwardDiff.jacobian!(DiffResults.hessian(c.result), c, DiffResults.gradient(c.result), x, c.jacobian_config, Val{false}())
    return DiffResults.hessian(c.result)
end

"""
Force (re-)evaluation of the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function NLSolversBase.value!!(obj::ForwardDiffDifferentiable, x)
    # obj.f_calls .+= 1
    copyto!(obj.x_f, x)
    DiffResults.value!(obj, f(obj, x) )
end
"""
Evaluates the objective value at `x`.

Returns `f(x)`, but does *not* store the value in `obj.F`
"""
function NLSolversBase.value(obj::ForwardDiffDifferentiable, x)
    if x != obj.x_f
        # obj.f_calls .+= 1
        NLSolversBase.value!!(obj, x)
    end
    NLSolversBase.value(obj)
end
"""
Evaluates the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function NLSolversBase.value!(obj::ForwardDiffDifferentiable, x)
    if x != obj.x_f
        NLSolversBase.value!!(obj, x)
    end
    NLSolversBase.value(obj)
end

"""
Evaluates the gradient value at `x`

This does *not* update `obj.DF`.
"""
function NLSolversBase.gradient(obj::ForwardDiffDifferentiable, x)
    DF = NLSolversBase.gradient(obj)
    if x != obj.x_df
        tmp = copy(DF)
        NLSolversBase.gradient!!(obj, x)
        @inbounds for i ∈ eachindex(tmp)
            tmp[i], DF[i] = DF[i], tmp[i]
        end
        return tmp
    end
    DF
end
"""
Evaluates the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function NLSolversBase.gradient!(obj::ForwardDiffDifferentiable, x)
    if x != obj.x_df
        NLSolversBase.gradient!!(obj, x)
    end
    nothing
end
"""
Force (re-)evaluation of the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function NLSolversBase.gradient!!(obj::ForwardDiffDifferentiable, x)
    # obj.df_calls .+= 1
    copyto!(obj.x_df, x)
    df(obj, x)
end

function NLSolversBase.value_gradient!(obj::ForwardDiffDifferentiable, x)
    if x != obj.x_f && x != obj.x_df
        NLSolversBase.value_gradient!!(obj, x)
    elseif x != obj.x_f
        NLSolversBase.value!!(obj, x)
    elseif x != obj.x_df
        NLSolversBase.gradient!!(obj, x)
    end
    value(obj)
end
function NLSolversBase.value_gradient!!(obj::ForwardDiffDifferentiable, x)
    # obj.f_calls .+= 1
    # obj.df_calls .+= 1
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    DiffResults.value!(obj, fdf(obj, x))
end

function NLSolversBase.hessian!(obj::ForwardDiffDifferentiable, x)
    if x != obj.x_h
        hessian!!(obj, x)
    end
end
function NLSolversBase.hessian!!(obj::ForwardDiffDifferentiable, x)
    # obj.h_calls .+= 1
    copyto!(obj.x_h, x)
    hessian!(obj.config, x)
end

# Getters are without ! and accept only an objective and index or just an objective
# "Get the most recently evaluated objective value of `obj`."
# "Get the most recently evaluated gradient of `obj`."
# "Get the most recently evaluated Jacobian of `obj`."
# jacobian(obj::AbstractObjective) = obj.DF
# "Get the `i`th element of the most recently evaluated gradient of `obj`."
# "Get the most recently evaluated Hessian of `obj`"


# function NLSolversBase._clear_f!(d::ForwardDiffDifferentiable)
#     # d.f_calls .= 0
#     if typeof(d.F) <: AbstractArray
#         d.F .= eltype(d.F)(NaN)
#     else
#         d.F = typeof(d.F)(NaN)
#     end
#     d.x_f .= eltype(d.x_f)(NaN)
#     d.x_f
# end

# function NLSolversBase._clear_df!(d::ForwardDiffDifferentiable)
#     d.df_calls .= 0
#     d.DF .= eltype(d.DF)(NaN)
#     d.x_df .= eltype(d.x_df)(NaN)
#     d.x_df
# end

# function NLSolversBase._clear_h!(d::ForwardDiffDifferentiable)
#     d.h_calls .= 0
#     d.H .= eltype(d.H)(NaN)
#     d.x_h .= eltype(d.x_h)(NaN)
#     d.x_h
# end

# function NLSolversBase.clear!(d::ForwardDiffDifferentiable)
#     _clear_f!(d)
#     _clear_df!(d)
#     _clear_h!(d)
#     nothing
# end

struct LightOptions{T}
    x_tol::T
    f_tol::T
    g_tol::T
end
LightOptions() = LightOptions(0.0,0.0,1e-8)

function optimize_light(d::D, initial_x::Tx, method::M,
                  options::Union{Options,LightOptions} = LightOptions(),
                  state = Optim.initial_state(method, options, d, initial_x)) where {D<:NLSolversBase.AbstractObjective, M<:Optim.AbstractOptimizer, Tx <: AbstractArray}

    # stopped = false, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased = false, false, false
    # counter_f_tol = 0

    g_converged = Optim.initial_convergence(d, state, method, initial_x, options)
    converged = g_converged

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0


    while !converged && iteration < options.iterations
        iteration += 1

        Optim.update_state!(d, state, method) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
        Optim.update_g!(d, state, method) # TODO: Should this be `update_fg!`?
        x_converged, f_converged,
        g_converged, converged, f_increased = Optim.assess_convergence(state, d, options)
        # For some problems it may be useful to require `f_converged` to be hit multiple times
        # TODO: Do the same for x_tol?
        # counter_f_tol = f_converged ? counter_f_tol+1 : 0
        # converged = converged | (counter_f_tol > options.successive_f_tol)

        !converged && Optim.update_h!(d, state, method) # only relevant if not converged

        # if tracing
        #     # update trace; callbacks can stop routine early by returning true
        #     stopped_by_callback = trace!(tr, d, state, iteration, method, options)
        # end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        # stopped_by_time_limit = time()-t0 > options.time_limit ? true : false
        # f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
        # g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false
        # h_limit_reached = options.h_calls_limit > 0 && h_calls(d) >= options.h_calls_limit ? true : false

        # if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
        #     stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
        #     stopped = true
        # end
    end # while

    # after_while!(d, state, method, options)

    # we can just check minimum, as we've earlier enforced same types/eltypes
    # in variables besides the option settings
    # T = typeof(options.f_tol)
    f_incr_pick = f_increased && !options.allow_f_increases
    # x_absc = x_abschange(state)
    # minf = pick_best_f(f_incr_pick, state, d)
    return Optim.pick_best_x(f_incr_pick, state), Optim.pick_best_f(f_incr_pick, state, d)#,  iteration
end
