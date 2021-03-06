@inline convergenceλ(x, abstol, reltol) = max(abstol, max(one(real(x)), norm(x)) * reltol)

function no_convergence(xn0, xn1, fxn1, abstol, reltol)
    λ = convergenceλ(xn1, abstol, reltol)
    (norm(fxn1) <= λ || (check_approx(xn1, xn0, abstol, reltol) && norm(fxn1) <= cbrt(λ))) ? false : true
end #on ifelse: isn't there going to be a branch anyway when the calling program checks whether this returned true/faslse?
function convergence(xn0, xn1, fxn1, abstol, reltol)
    λ = convergenceλ(xn1, abstol, reltol)
    (norm(fxn1) <= λ || (check_approx(xn1, xn0, abstol, reltol) && norm(fxn1) <= cbrt(λ))) ? true : false
end #on ifelse: isn't there going to be a branch anyway when the calling program checks whether this returned true/faslse?


@inline function initial_conditions(ap::AbstractAsymptoticPosterior, i = profile_ind(ap))
    @inbounds begin
        fx0 = rstar(ap)
        x0 = mode(ap, i)
        x1 = x0 + fx0 * std_estimates(ap,i) # inverse slope
    end
    debug_rootsearch() && @show fx0, x0, x1
    fx1, s = fdf_adjrstar_p(ap, x1, i, Val{true}())
    x0, fx0, x1, fx1, s

end

function linear_search(ap::AbstractAsymptoticPosterior{P,T}, i = profile_ind(ap)) where {P,T}
    x0, fx0, x1, fx1, s = initial_conditions(ap, i)
    ϵ = eps(max(abs(x0),abs(x1)))
    abstol, reltol = √ϵ, ∛ϵ
    # if norm(fx1) <= convergenceλ(x1, abstol, reltol)
    #     return x1#, fx1
    # elseif signbit(fx0) != signbit(fx1)
    #     return custom_bisection(ap, x1, x0, fx1, fx0, true, cbrt(eps()))
    # end
    norm(fx1) <= convergenceλ(x1, abstol, reltol) && return x1

    debug_rootsearch() && @show ((x0 - x1) / (fx1 - fx0), 1 / s)
    debug_rootsearch() && @show x1 + fx1 / s
    debug_rootsearch() && @show x1 + (fx1 * (x0 - x1)) / (fx1 - fx0)
    # x1, x0 = x1 + fx1 * (x0 - x1) / (fx1 - fx0), x1
    x1, x0 = x1 - fx1 / s, x1
    debug_rootsearch() && @show x1

    debug_rootsearch() && @show rstar(ap)
    fx0 = fx1
    fx1, s = fdf_adjrstar_p(ap, x1, i, Val{false}())
    debug_rootsearch() && @show fx1, ap(x1, i)
    # fx1, fx0 = ap(x1, i), fx1

    not_converged = no_convergence(x0, x1, fx1, abstol, reltol)
    # @show not_converged

    # Keep using quadratic approximations until either convergence or signs flip.
    while not_converged && signbit(fx0) == signbit(fx1)
        # x1, x0 = x1 + fx1 * (x0 - x1) / (fx1 - fx0), x1
        x1, x0 = x1 - fx1 / s, x1
        # fx1, fx0 = ap(x1, i), fx1
        fx0 = fx1
        fx1, s = fdf_adjrstar_p(ap, x1, i, Val{false}())
        debug_rootsearch() && @show (fx1, fx0)
        not_converged = no_convergence(x0, x1, fx1, abstol, reltol)
    end
    if not_converged # If signs flipped, switch to Brent
        debug_rootsearch() && @show (x0, fx0, ap(x0, i))
        debug_rootsearch() && @show (x1, fx1, ap(x1, i))
        # reset_state!(ap.state, x1, x0, fx1, fx0)
        # x1 = ap.state.xn1
        x1 = custom_bisection(ap, x1, x0, Φ⁻¹(1 - α(ap) - fx1) + rstar(ap), Φ⁻¹(1 - α(ap) - fx0) + rstar(ap), abstol)
        # fx1 = ap.state.fx1
    end

    x1#, fx1
end

function update(x0, x1, x2, fx0, fx1, fx2)

    # δx01 = (x0 - x1)
    # δx02 = (x0 - x2)
    # δx12 = (x1 - x2)
    # denom = δx01*δx02*δx12

    δx01 = fx2 * (x0 - x1)
    δx02 = fx1 * (x0 - x2)
    δx12 = fx0 * (x1 - x2)
    a = (δx12         - δx02         + δx01)#/denom
    b = (δx12*(x1+x2) - δx02*(x0+x2) + δx01*(x0+x1))/2a#/denom
    c = (δx12*x1*x2   - δx02*x0*x2   + δx01*x0*x1)#/denom


    discriminant = abs2(b) - c/a
    if discriminant > 0 # Would projecting whether the upper or lower root is correct be faster?
        root_discrim = sqrt(discriminant)
        r_u = b + root_discrim
        r_l = b - root_discrim
        x0 = ifelse(norm(r_u - x2) < norm(r_l-x2), r_u, r_l)
    else
        x0 = x2 + fx2 * δx12 / (fx2-fx1)
    end

    # a = (δx12         - δx02         + δx01)#/denom
    # b = (δx12*(x1+x2) - δx02*(x0+x2) + δx01*(x0+x1))/2#/denom
    # c = (δx12*x1*x2   - δx02*x0*x2   + δx01*x0*x1)#/denom


    # discriminant = abs2(b) - c*a
    # if discriminant > 0 # Would projecting whether the upper or lower root is correct be faster?
    #     root_discrim = sqrt(discriminant)
    #     r_u = (b + root_discrim) / a
    #     r_l = (b - root_discrim) / a
    #     x0 = ifelse(norm(r_u - x2) < norm(r_l-x2), r_u, r_l)
    # else
    #     x0 = x2 + fx2 * δx12 / (fx2-fx1)
    # end

    x1, x2, x0
end


function quadratic_search(ap::AbstractAsymptoticPosterior{P,T}, i = profile_ind(ap)) where {P,T}
    x0, fx0, x1, fx1, s = initial_conditions(ap, i)

    abstol, reltol = 4eps(T), cbrt(eps(T))
    if norm(fx1) <= convergenceλ(x1, abstol, reltol)
        return x1#, fx1
    end

    # s = -s
    δf = fx1 - fx0
    δx = x1 - x0
    # sδx = δx*s
    # a = δf + sδx
    # b = (abs2(x1) - abs2(x0))*s + 2x0*δf
    # c = fx1*abs2(x0) + fx0*abs2(x1) - x0*x1*(2fx0 - sδx)
    # # @show (a,b,c)
    # # println("$x0 $c $b $a")
    # # println("$x1 $c $b $a")
    # # @show (x0, x1, fx0, fx1, s)

    # discriminant = abs2(b) -4a*c
    # if discriminant > 0
    #     x2 = (b-sqrt(discriminant)) / 2a
    # else #If there are no solutions to the quadratic problem, we will take a linear step.
        x2 = x1 - fx1 * δx / δf
    # end
    fx2 = ap(x2, i)
    not_converged = no_convergence(x1, x2, fx2, abstol, reltol)

    # Keep using quadratic approximations until either convergence or signs flip.
    while not_converged && signbit(fx2) == signbit(fx1)
        x0,  x1,  x2  = update(x0, x1, x2, fx0, fx1, fx2)
        fx0, fx1, fx2 = fx1, fx2, ap(x2, i)
        not_converged = no_convergence(x1, x2, fx2, abstol, reltol)
    end
    if not_converged # If signs flipped, switch to FalsePosition
        # reset_state!(ap.state, x2, x1, fx2, fx1)
        # x2 = ap.state.xn1
        x2 = custom_bisection(ap, x2, x1, fx2, fx1, true, cbrt(eps()))
        # fx2 = ap.state.fx1
    end

    x2#, fx2
end
