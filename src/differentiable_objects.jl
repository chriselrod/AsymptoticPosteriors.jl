
# Could these be Base.@pure ?
@generated Chunk(::Val{N}) where N = ForwardDiff.Chunk(N)
@generated ValP1(::Val{N}) where N = Val{N+1}()
@generated ValM1(::Val{N}) where N = Val{N-1}()

abstract type ForwardDiffDifferentiable <: NLSolversBase.AbstractObjective end


struct Configuration{T,T2,V,ND,DJ,DG,DG,F}
    f::F
    result::DiffResults.MutableDiffResult{V,Tuple{Vector{V},Matrix{V}}}
    inner_result::DiffResults.MutableDiffResult{Dual{T,V,ND},Tuple{Vector{Dual{T,V,ND}}}}
    jacobian_config::ForwardDiff.JacobianConfig{T,V,ND,DJ}
    gradient_config::ForwardDiff.GradientConfig{T,Dual{T,V,ND},ND,DG}
    gconfig::ForwardDiff.GradientConfig{T2,V,ND,DG}
end

struct LeanDifferentiable{{N,T,A<:AbstractArray{T},C} <: ForwardDiffDifferentiable
    x_f::A # x used to evaluate f (stored in F)
    x_df::A # x used to evaluate df (stored in DF)
    x_h::A #??
    config::C
    # f_calls::Vector{Int}
    # df_calls::Vector{Int}
end

# struct ProfileDifferentiable{N,T,A<:AbstractArray{T},C} <: ForwardDiffDifferentiable
#     x_f::Vector{T} # x used to evaluate f (stored in F)
#     x_df::Vector{T} # x used to evaluate df (stored in DF)
#     x_h::Vector{T} #??
#     config::C
#     x::A
#     fixed_ind::RefValue{Int}
#     fixed_val::RefValue{T}
#     # f_calls::Vector{Int}
#     # df_calls::Vector{Int}
# end

function Configuration(f::F, x::AbstractArray{T}, ::Val{N}) where {F,T,N}

    result = DiffResults.HessianResult(x)

    chunk = Chunk(Val{N}())
    tag = ForwardDiff.Tag(f, T)
    jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
    gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)

    Configuration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
end
LeanDifferentiable(f::F, ::Val{N}) where {F,N} = LeanDifferentiable(f, Vector{Float64}(undef, N), Val{N}())
function LeanDifferentiable(f::F, x::AbstractArray{T}, ::Val{N}) where {F,T,N}
    LeanDifferentiable(x, similar(x), similar(x), Configuration(f, x, Val{N}()), Val{N}())
end
function LeanDifferentiable(x_f::A,x_df::A,x_h::A,config::C,::Val{N}) where {T,A<:AbstractArray{T},C,N}
    LeanDifferentiable{N,T,A,C}(x_f, x_df, x_h, config)
end

# ProfileDifferentiable(f::F, ::Val{N}) where {F,N} = ProfileDifferentiable(f, Vector{Float64}(undef, N), Val{N}())
# function ProfileDifferentiable(f::F, x::AbstractArray{T}, ::Val{N}) where {F,T,N}
#     x_f = Vector{T}(undef, N-1)
#     ProfileDifferentiable(x_f, similar(x_f), similar(x_f), Configuration(f, x_f, ValM1(Val{N}())), x, Ref(N), Ref{T}(),::Val{N})
# end
# function ProfileDifferentiable(x_f::Vector{T},x_df::Vector{T},x_h::Vector{T}, config::C, x,::A i::RefValue{Int}, v::RefValue{T},::Val{N}) where {T,A<:AbstractArray{T},C,N}
#     ProfileDifferentiable{N,T,A,C}(x_f, x_df, x_h, config, x, i, v)
# end



DiffResults.value!(obj::ForwardDiffDifferentiable, x::Real) = DiffResults.value!(obj.config.result, x)

NLSolversBase.value(obj::ForwardDiffDifferentiable) = DiffResults.value(obj.config.result)
NLSolversBase.gradient(obj::ForwardDiffDifferentiable) = DiffResults.gradient(obj.config.result)
NLSolversBase.gradient(obj::ForwardDiffDifferentiable, i::Integer) = DiffResults.gradient(obj.config.result)[i]
NLSolversBase.hessian(obj::ForwardDiffDifferentiable) = DiffResults.hessian(obj.config.result)


f(obj::LeanDifferentiable, x) = obj.config.f(x)
# function f(obj::ProfileDifferentiable, x)


# end

function df(obj::ForwardDiffDifferentiable, x)
    ForwardDiff.gradient!(gradient(obj), obj.config.f, x, obj.config.gconfig, Val{false}())
end
function fdf(obj::ForwardDiffDifferentiable, x)
    obj.config.result.derivs = (gradient(obj), hessian(obj))
    ForwardDiff.gradient!(obj.config.result, obj.config.f, x, obj.config.gconfig, Val{false}())
    DiffResults.value(obj.config.result)
end

# function Configuration(f::F, x::AbstractArray{V}, chunk::Chunk = Chunk(x), tag = Tag(f, V)) where {F,V}
#     result = DiffResults.HessianResult(x)
#     jacobian_config = ForwardDiff.JacobianConfig((f,gradient), DiffResults.gradient(result), x, chunk, tag)
#     gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
#     inner_result = DiffResults.DiffResult(zero(eltype(jacobian_config.duals[2])), jacobian_config.duals[2])
#     gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)
#     Configuration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
# end

function (c::Configuration)(y, z)
    c.inner_result.derivs = (y,) #Already true?
    ForwardDiff.gradient!(c.inner_result, c.f, z, c.gradient_config, Val{false}())
    DiffResults.value!(c.result, ForwardDiff.value(DiffResults.value(c.inner_result)))
    y
end
function hessian!(c::Configuration, x::AbstractArray) where {T,CHK}
    ForwardDiff.jacobian!(DiffResults.hessian(c.result), c, DiffResults.gradient(c.result), x, c.jacobian_config, Val{false}())
    DiffResults.hessian(c.result)
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
    NLSolversBase.gradient(obj)
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

struct LightOptions{T}
    x_tol::T
    f_tol::T
    g_tol::T
end
LightOptions() = LightOptions(0.0,0.0,1e-8)



# mutable struct BFGSState{Tx, Tm, T,G} <: AbstractOptimizerState
#     x::Tx
#     x_previous::Tx
#     g_previous::G
#     f_x_previous::T
#     dx::Tx
#     dg::Tx
#     u::Tx
#     invH::Tm
#     s::Tx
#     @add_linesearch_fields()
# end
function initial_state!(state, method::BFGS, d, initial_x::AbstractArray{T}) where T
    n = length(initial_x)
    copyto!(state.x, initial_x)
    Optim.retract!(method.manifold, initial_x)
    NLSolversBase.value_gradient!!(d, initial_x)
    Optim.project_tangent!(method.manifold, NLSolversBase.gradient(d), initial_x)
    copyto!(state.g_previous, NLSolversBase.gradient(d))
    method.initial_invH(state.invH, initial_x)
    state.alpha = one(T)
    nothing
end
# function initial_state!(state, method::BFGS, d, initial_x::AbstractArray{T}) where T
#     n = length(initial_x)
#     copyto!(state.x, initial_x)
#     Optim.retract!(method.manifold, initial_x)
#     value_gradient!!(d, initial_x)
#     project_tangent!(method.manifold, gradient(d), initial_x)
#     copyto!(state.g_previous, gradient(d))
#     method.initial_invH(state.invH, initial_x)
#     state.alpha = one(T)
#     nothing
# end

function uninitialized_state(initial_x::AbstractArray{T}) where T
    Optim.BFGSState(similar(initial_x), # Maintain current state in state.x
        similar(initial_x), # Maintain previous state in state.x_previous
        similar(initial_x), # Store previous gradient in state.g_previous
        T(NaN), # Store previous f in state.f_x_previous
        similar(initial_x), # Store changes in position in state.dx
        similar(initial_x), # Store changes in gradient in state.dg
        similar(initial_x), # Buffer stored in state.u
        Matrix{T}(undef, length(x), length(x)), # Store current invH in state.invH
        similar(initial_x), # Store current search direction in state.s
        T(NaN),            # Keep track of previous descent value ⟨∇f(x_{k-1}), s_{k-1}⟩
        similar(initial_x), # Buffer of x for line search in state.x_ls
        one(T))
end

function optimize_light(d::D, initial_x::Tx, method::M,
                  options::Union{Options,LightOptions} = LightOptions(),
                  state = Optim.initial_state(method, options, d, initial_x)) where {D<:NLSolversBase.AbstractObjective, M<:Optim.AbstractOptimizer, Tx <: AbstractArray}

    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased = false, false, false

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
        !converged && Optim.update_h!(d, state, method) # only relevant if not converged
    end # while

    f_incr_pick = f_increased && !options.allow_f_increases
    return Optim.pick_best_x(f_incr_pick, state), Optim.pick_best_f(f_incr_pick, state, d)
end
