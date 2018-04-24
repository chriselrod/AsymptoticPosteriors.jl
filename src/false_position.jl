# Taken and modified from Roots.jl
# https://github.com/JuliaMath/Roots.jl/blob/master/LICENSE.md
# The attached Liscense Agreement:

# The MIT License (MIT) Copyright (c) 2013 John C. Travers

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

abstract type UnivariateZeroMethod end
abstract type AbstractBisection <: UnivariateZeroMethod end

struct FalsePosition{R} <: AbstractBisection end
FalsePosition() = FalsePosition{:anderson_bjork}()

function find_zero(f::F, x0::Tuple{T,S}, method::FalsePosition) where {T<:Real,S<:Real,F}
    find_zero(f, adjust_bracket(x0), method)
end

function find_zero!(state, f::F, x0::Tuple{T,S}, method::FalsePosition) where {T<:Real,S<:Real,F}
    find_zero!(state, f, adjust_bracket(x0), method)
end
function find_zero(f::F, x0::Tuple{T,T}, method::FalsePosition) where {T<:AbstractFloat,F}
    find_zero(f, x0, method, UnivariateZeroOptions(T))
end

function find_zero!(state, f::F, x0::Tuple{T,T}, method::FalsePosition) where {T<:AbstractFloat ,F}
    find_zero!(state, f, x0, method, UnivariateZeroOptions(T))
end


function adjust_bracket(x0)
    u, v = float.(x0)
    if u > v
        u, v = v, u
    end

    if isinf(u)
        u = nextfloat(u)
    end
    if isinf(v)
        v = prevfloat(v)
    end
    u, v
end


# struct UnivariateZeroProblem{T<:AbstractFloat, F}
#     fs::F
#     x0::Tuple{T,T}
# end
struct UnivariateZeroOptions{T}
    xabstol::T
    xreltol::T
    abstol::T
    reltol::T
    maxevals::Int
    maxfnevals::Int
end


function UnivariateZeroOptions(::Type{T} = Float64, xabstol=zero(T), xreltol=eps(T),
                               abstol=4*eps(T), reltol=4*eps(T),
                               maxevals=40, maxfnevals=typemax(Int) ) where T
    UnivariateZeroOptions{T}(xabstol, xreltol, abstol,
                                    reltol, maxevals, maxfnevals)
end


function find_zero!(state, fs::F, x0, method::UnivariateZeroMethod, options::UnivariateZeroOptions) where F
    reset_state!(state, method, fs, x0)
    find_zero!(method, fs, state, options)#, [state.xn1], [state.fxn1])
end
function find_zero(fs::F, x0, method::UnivariateZeroMethod, options::UnivariateZeroOptions) where F
    state = UnivariateZeroStateBase(method, fs, x0)
    find_zero!(state, fs, method, options)#, [state.xn1], [state.fxn1])
end


function reset_state!(state, method::AbstractBisection, fs::F, x::Tuple{T,T}) where {T <: Real, F}
    x0, x2 = x
    reset_state!(state, x0, x2, fs(x0), fs(x2))
end

abstract type UnivariateZeroState{T,S} end

mutable struct UnivariateZeroStateBase{T,S,AS<:AbstractString} <: UnivariateZeroState{T,S}
    # xn2::T
    xn1::T
    xn0::T
    # fxn2::S
    fxn1::S
    fxn0::S
    steps::Int
    fnevals::Int
    stopped::Bool             # stopped, butmay not have converged
    x_converged::Bool         # converged via |x_n - x_{n-1}| < ϵ
    f_converged::Bool         # converged via |f(x_n)| < ϵ
    convergence_failed::Bool
    message::AS
end
function UnivariateZeroStateBase(method::AbstractBisection, fs::F, x::Tuple{T,T}) where {T <: Real, F}
    x0, x2 = x
    y0 = fs(x0)
    y2 = fs(x2)

    # @assert sign(y0) != sign(y2)

    UnivariateZeroStateBase(x0, x2,
                            y0, y2,
                            0, 2,
                            false, false, false, false, "")
end

function reset_state!(state::UnivariateZeroState{T,S}, x0::T, x2::T, y0::S, y2::S) where {T,S}
    state.xn1 = x0
    state.xn0 = x2
    state.fxn1 = y0
    state.fxn0 = y2
    state.steps = 0
    state.fnevals = 2
    state.stopped = false
    state.x_converged = false
    state.f_converged = false
    state.convergence_failed = false
    # state.message = ""
    state
end

function find_zero!(state::UnivariateZeroState, fs::F, method::UnivariateZeroMethod, options::UnivariateZeroOptions) where F
    ## XXX Should just deprecate this in favor of FalsePosition method XXX
    while true
        
        val = assess_convergence(state, options)

        if val
            if state.stopped && !(state.x_converged || state.f_converged)
                ## stopped is a heuristic, there was an issue with an approximate derivative
                ## say it converged if pretty close, else say convergence failed.
                ## (Is this a good idea?)
                xstar, fxstar = state.xn1, state.fxn1
                if abs(fxstar) <= (options.abstol)^(2/3)
                    msg = "Algorithm stopped early, but |f(xn)| < ϵ^(2/3), where ϵ = abstol"
                    state.message = state.message == "" ? msg : state.message * "\n\t" * msg
                    state.f_converged = true
                else
                    state.convergence_failed = true
                end
            end
                
            if state.x_converged || state.f_converged
                # options.verbose && show_trace(fs, state, xns, fxns, method)
                return state#.xn1
            end

            if state.convergence_failed
                # options.verbose && show_trace(fs, state, xns, fxns, method)
                throw("Stopped at: xn = $(state.xn1)")
            end
        end

        update_state(method, fs, state, options)
        # if options.verbose
        #     push!(xns, state.xn1)
        #     push!(fxns, state.fxn1)
        # end

    end
end


function update_state(method::FalsePosition, fs::F, o, options) where F

    # fs
    a, b =  o.xn0, o.xn1

    fa, fb = o.fxn0, o.fxn1

    lambda = fb / (fb - fa)
    tau = 1e-10                   # some engineering to avoid short moves
    if !(tau < norm(lambda) < 1-tau)
        lambda = 1/2
    end
    x = b - lambda * (b-a)        
    fx = fs(x)
    o.fnevals += 1
    o.steps += 1

    if iszero(fx)
        o.xn1 = x
        o.fxn1 = fx
        return
    end

    if sign(fx)*sign(fb) < 0
        a, fa = b, fb
    else
        fa = reduction(method, fa, fb, fx)
    end
    b, fb = x, fx

    #xn0 are old, xn1 are new.
    o.xn0, o.xn1 = a, b 
    o.fxn0, o.fxn1 = fa, fb
    
    nothing
end


# the 12 reduction factors offered by Galadino
const galdino = Dict{Union{Int,Symbol},Function}(:1 => (fa, fb, fx) -> fa*fb/(fb+fx),
                                            :pegasus => (fa, fb, fx) -> fa*fb/(fb+fx),
                                           :2 => (fa, fb, fx) -> (fa - fb)/2,
                                           :3 => (fa, fb, fx) -> (fa - fx)/(2 + fx/fb),
                                           :4 => (fa, fb, fx) -> (fa - fx)/(1 + fx/fb)^2,
                                           :5 => (fa, fb, fx) -> (fa -fx)/(1.5 + fx/fb)^2,
                                           :6 => (fa, fb, fx) -> (fa - fx)/(2 + fx/fb)^2,
                                           :7 => (fa, fb, fx) -> (fa + fx)/(2 + fx/fb)^2,
                                           :8 => (fa, fb, fx) -> fa/2,
                                           :illinois => (fa, fb, fx) -> fa/2,
                                           :9 => (fa, fb, fx) -> fa/(1 + fx/fb)^2,
                                           :10 => (fa, fb, fx) -> (fa-fx)/4,
                                           :11 => (fa, fb, fx) -> fx*fa/(fb+fx),
                                           :12 => (fa, fb, fx) -> (fa * (1-fx/fb > 0 ? 1-fx/fb : 1/2)),
                                           :anderson_bjork => (fa, fb, fx) -> (fa * (1-fx/fb > 0 ? 1-fx/fb : 1/2))
)
# give common names
# for (nm, i) in [(:pegasus, 1), (:illinois, 8), (:anderson_bjork, 12)]
#     galdino[nm] = galdino[i]
# end
@generated function reduction(method::FalsePosition{R}, fa, fb, fx) where R
    f = galdino[R]
    quote
        $Expr(:meta, :inline)
        $f(fa, fb, fx)
    end
end


function assess_convergence(state, options)

    xn0, xn1 = state.xn0, state.xn1
    fxn0, fxn1 = state.fxn0, state.fxn1

    
    if (state.x_converged || state.f_converged)
        return true
    end

    if state.steps > options.maxevals
        state.stopped = true
        state.message = "too many steps taken."
        return true
    end

    if state.fnevals > options.maxfnevals
        state.stopped = true
        state.message = "too many function evaluations taken."
        return true
    end

    if isnan(xn1)
        state.convergence_failed = true
        state.message = "NaN produced by algorithm"
        return true
    end
    
    if isinf(fxn1)
        state.convergence_failed = true
        state.message = "Inf produced by algorithm"
        return true
    end

    λ = convergenceλ(xn1, options)
    
    #is fxn1 close enough to 0?
    if  norm(fxn1) <= λ
        state.f_converged = true
        return true
    end

    # Or are xn1 and xn0 close enough to one another?
    if check_approx(xn1, xn0, options.xreltol, options.xabstol) && norm(fxn1) <= cbrt(λ)
        state.x_converged = true
        return true
    end


    if state.stopped
        if state.message == ""
            error("no message? XXX debug this XXX")
        end
        return true
    end

    return false
end

@inline convergenceλ(x, options) = max(options.abstol, max(one(real(x)), norm(x)) * options.reltol) 
@inline check_approx(x, y, rtol, atol) = norm(x-y) <= atol + rtol*max(norm(x), norm(y))
# function check_approx(x, y, rtol, atol)
#     @show norm(x-y)
#     @show atol + rtol*max(norm(x), norm(y))
#     norm(x-y) <= atol + rtol*max(norm(x), norm(y))
# end


