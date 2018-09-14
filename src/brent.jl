### Credit to djsegal
### https://discourse.julialang.org/t/is-there-a-faster-bisection-root-solver-that-uses-atol/12658/21

# f, cur_a, cur_b, cur_f_a, cur_f_b  = ap, x1, x0, fx1, fx0;
# cur_flag, abstol = true, cbrt(eps())
# cur_c, cur_f_c, cur_d, cur_f_d, bad_streak = NaN,NaN,NaN,NaN,0

@inline check_approx(x, y, atol, rtol) = norm(x-y) <= atol + rtol*max(norm(x), norm(y))
@inline check_approx(x, y, atol) = norm(x-y) <= atol

function custom_bisection(f, cur_a::T, cur_b::T, cur_f_a::T, cur_f_b::T, abstol=sqrt(eps(max(cur_a,cur_b))) ) where T
  # cur_s = zero(T)
  local cur_s::T
  cur_d, cur_f_d = T(NaN), T(NaN)
  cur_flag = true
  cur_c = middle(cur_a, cur_b)
  cur_f_c = f(cur_c)
  # for iter = 1:100
  for iteration ∈ 1:100
    init_a, init_b = cur_a, cur_b
    init_f_a, init_f_b = cur_f_a, cur_f_b

    if abs(cur_f_a) < abs(cur_f_b)
      cur_a, cur_b = cur_b, cur_a
      cur_f_a, cur_f_b = cur_f_b, cur_f_a
    end

    check_approx(cur_f_a, 0.0, abstol) && return cur_a
    check_approx(cur_f_b, 0.0, abstol) && return cur_b
    check_approx(cur_f_c, 0.0, abstol) && return cur_c
    check_approx(cur_f_d, 0.0, abstol) && return cur_d

    check_approx(cur_a, cur_b, abstol) && return cur_c
    check_approx(cur_f_a, cur_f_b, abstol) && return NaN

    is_good_a = isfinite(cur_f_a)
    is_good_b = isfinite(cur_f_b)
    is_good_c = isfinite(cur_f_c)
    is_good_d = isfinite(cur_f_d)

    ( is_good_a && is_good_b && cur_f_a * cur_f_b > 0 ) && return NaN

    ( is_good_a || is_good_b ) || return NaN

    if !(is_good_a && is_good_b && is_good_c)
      if is_good_d && ( check_approx(cur_c, cur_a, abstol) || check_approx(cur_c, cur_b, abstol) )
        cur_g = cur_d
        cur_f_g = cur_f_d
      else
        cur_g = cur_c
        cur_f_g = cur_f_c
      end

      if is_good_a
        cur_root = custom_bisection(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol)
        isnan(cur_root) &&
          ( cur_root = custom_bisection(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol) )
        return cur_root
      else
        cur_root = custom_bisection(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol)
        isnan(cur_root) &&
          ( cur_root = custom_bisection(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol) )
        return cur_root
      end
    end

    # cur_d = NaN
    # cur_s = zero(T)

    if !check_approx(cur_f_a, cur_f_c, abstol) && !check_approx(cur_f_b, cur_f_c, abstol)
      cur_term_1 = cur_f_b / ( cur_f_a - cur_f_c )
      cur_term_2 = cur_f_c / ( cur_f_a - cur_f_b )
      cur_term_3 = cur_f_a / ( cur_f_b - cur_f_c )

      cur_s = cur_a * cur_term_1 * cur_term_2
      cur_s -= cur_b * cur_term_2 * cur_term_3
      cur_s += cur_c * cur_term_3 * cur_term_1
    else
      cur_s = cur_b - cur_f_b * ( cur_b - cur_a ) / ( cur_f_b - cur_f_a )
    end

    s_bound = ( T(3) * cur_a + cur_b ) * T(0.25)
    cur_bool = ( cur_s < cur_b && cur_s < s_bound  ) || ( cur_s > cur_b && cur_s > s_bound )

    if cur_flag
      cur_diff = abs( cur_b - cur_c )
    else
      cur_diff = abs( cur_d - cur_c )
    end

    cur_bool |= abs( cur_s - cur_b ) >= ( cur_diff * T(0.5) )
    cur_bool |= check_approx(cur_diff, 0.0, abstol)

    cur_flag = cur_bool
    cur_bool && ( cur_s = middle(cur_a, cur_b) )

    cur_d = cur_c
    cur_c = cur_b

    cur_f_d = cur_f_c
    cur_f_c = cur_f_b

    cur_f_s = f(cur_s)

    if cur_f_a * cur_f_s < zero(T)
      cur_b = cur_s
      cur_f_b = cur_f_s
    else
      cur_a = cur_s
      cur_f_a = cur_f_s
    end

    # is_bad_f_a = check_approx(cur_f_a, init_f_a, abstol)
    # is_bad_f_b = check_approx(cur_f_b, init_f_b, abstol)
    #
    # is_bad_f_a |= abs(cur_f_a) > abs(init_f_a)
    # is_bad_f_b |= abs(cur_f_b) > abs(init_f_b)
    #
    # if ( is_bad_f_a && is_bad_f_b )
    #   bad_streak += 1
    # else
    #   bad_streak = 0
    # end
    #
    # ( bad_streak > 4 ) && return NaN

    # is_unchanged = check_approx(cur_a, init_a, atol=2*eps())
    # is_unchanged &= check_approx(cur_b, init_b, atol=2*eps())
    #
    # if is_unchanged
    #   cur_c = middle(cur_a, cur_b)
    #   cur_f_c = f(cur_c)
    #   cur_flag = true
    #   # cur_d, cur_f_d = NaN, NaN
    # end
  end
  cur_s
end#cur_c::Number=NaN, cur_f_c::Number=NaN, cur_d::Number=NaN, cur_f_d::Number=NaN, bad_streak



function custom_bisection_old(f, cur_a::T, cur_b::T, cur_f_a::T, cur_f_b::T, cur_flag::Bool=true, abstol::Number=sqrt(eps(max(cur_a,cur_b))), cur_c::T=NaN, cur_f_c::T=NaN, cur_d::T=NaN, cur_f_d::T=NaN, bad_streak::Integer=0) where T
  init_a = cur_a
  init_b = cur_b

  # @show cur_a, cur_b

  init_f_a = cur_f_a
  init_f_b = cur_f_b

  isnan(cur_c) && ( cur_c = middle(cur_a, cur_b) )
  isnan(cur_f_c) && ( cur_f_c = f(cur_c) )

  if abs(cur_f_a) < abs(cur_f_b)
    cur_a, cur_b = cur_b, cur_a
    cur_f_a, cur_f_b = cur_f_b, cur_f_a
  end

  check_approx(cur_f_a, 0.0, abstol) && return cur_a
  check_approx(cur_f_b, 0.0, abstol) && return cur_b
  check_approx(cur_f_c, 0.0, abstol) && return cur_c
  check_approx(cur_f_d, 0.0, abstol) && return cur_d

  check_approx(cur_a, cur_b, abstol) && return cur_c
  check_approx(cur_f_a, cur_f_b, abstol) && return NaN

  is_good_a = isfinite(cur_f_a)
  is_good_b = isfinite(cur_f_b)
  is_good_c = isfinite(cur_f_c)
  is_good_d = isfinite(cur_f_d)

  ( is_good_a && is_good_b && cur_f_a * cur_f_b > 0 ) && return NaN

  ( is_good_a || is_good_b ) || return NaN

  if !(is_good_a && is_good_b && is_good_c)
    if is_good_d && ( check_approx(cur_c, cur_a, abstol) || check_approx(cur_c, cur_b, abstol) )
      cur_g = cur_d
      cur_f_g = cur_f_d
    else
      cur_g = cur_c
      cur_f_g = cur_f_c
    end

    # if is_bad_a
    #   cur_root = custom_bisection_old(f, cur_g, cur_b, cur_f_c, cur_f_b, abstol=abstol)
    #   isnan(cur_root) &&
    #     ( cur_root = custom_bisection_old(f, cur_a, cur_g, cur_f_a, cur_f_c, abstol=abstol) )
    #   return cur_root
    # else
    #   cur_root = custom_bisection_old(f, cur_a, cur_g, cur_f_a, cur_f_c, abstol=abstol)
    #   isnan(cur_root) &&
    #     ( cur_root = custom_bisection_old(f, cur_g, cur_b, cur_f_c, cur_f_b, abstol=abstol) )
    #   return cur_root
    # end
    if is_good_a
      cur_root = custom_bisection_old(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol)
      isnan(cur_root) &&
        ( cur_root = custom_bisection_old(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol) )
      return cur_root
    else
      cur_root = custom_bisection_old(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol)
      isnan(cur_root) &&
        ( cur_root = custom_bisection_old(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol) )
      return cur_root
    end
  end

  # cur_d = NaN
  # cur_s = zero(T)
  local cur_s::T

  if !check_approx(cur_f_a, cur_f_c, abstol) && !check_approx(cur_f_b, cur_f_c, abstol)
    cur_term_1 = cur_f_b / ( cur_f_a - cur_f_c )
    cur_term_2 = cur_f_c / ( cur_f_a - cur_f_b )
    cur_term_3 = cur_f_a / ( cur_f_b - cur_f_c )

    cur_s = cur_a * cur_term_1 * cur_term_2
    cur_s -= cur_b * cur_term_2 * cur_term_3
    cur_s += cur_c * cur_term_3 * cur_term_1
  else
    # cur_term = ( cur_b - cur_a )
    # cur_term /= ( cur_f_b - cur_f_a )
    #
    # cur_s = cur_b
    # cur_s -= cur_f_b * cur_term

    cur_s = cur_b - cur_f_b * ( cur_b - cur_a ) / ( cur_f_b - cur_f_a )
  end

  s_bound = ( T(3) * cur_a + cur_b ) * T(0.25)
  cur_bool = ( cur_s < cur_b && cur_s < s_bound  ) || ( cur_s > cur_b && cur_s > s_bound )

  if cur_flag
    cur_diff = abs( cur_b - cur_c )
  else
    cur_diff = abs( cur_d - cur_c )
  end

  cur_bool |= abs( cur_s - cur_b ) >= ( cur_diff * T(0.5) )
  cur_bool |= check_approx(cur_diff, 0.0, abstol)

  cur_flag = cur_bool
  cur_bool && ( cur_s = middle(cur_a, cur_b) )

  cur_d = cur_c
  cur_c = cur_b

  cur_f_d = cur_f_c
  cur_f_c = cur_f_b

  cur_f_s = f(cur_s)

  if cur_f_a * cur_f_s < zero(T)
    cur_b = cur_s
    cur_f_b = cur_f_s
  else
    cur_a = cur_s
    cur_f_a = cur_f_s
  end

  is_bad_f_a = check_approx(cur_f_a, init_f_a, abstol)
  is_bad_f_b = check_approx(cur_f_b, init_f_b, abstol)

  is_bad_f_a |= abs(cur_f_a) > abs(init_f_a)
  is_bad_f_b |= abs(cur_f_b) > abs(init_f_b)

  if ( is_bad_f_a && is_bad_f_b )
    bad_streak += 1
  else
    bad_streak = 0
  end

  ( bad_streak > 4 ) && return NaN

  is_unchanged = check_approx(cur_a, init_a, 2*eps())
  is_unchanged &= check_approx(cur_b, init_b, 2*eps())

  if is_unchanged
    cur_c = middle(cur_a, cur_b)
    cur_f_c = f(cur_c)

    # cur_root = custom_bisection_old(f, cur_a, cur_c, cur_f_a, cur_f_c, abstol=abstol, bad_streak=bad_streak)
    cur_root = custom_bisection_old(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol, NaN, NaN, NaN, NaN, bad_streak)
    isnan(cur_root) || return cur_root

    # cur_root = custom_bisection_old(f, cur_c, cur_b, cur_f_c, cur_f_b, abstol=abstol, bad_streak=bad_streak)
    # return cur_root
    return custom_bisection_old(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol, NaN, NaN, NaN, NaN, bad_streak)
  end

  # custom_bisection_old(
  #   f, cur_a, cur_b, cur_f_a, cur_f_b,
  #   cur_flag, abstol=abstol,
  #   cur_c=cur_c, cur_f_c=cur_f_c,
  #   cur_d=cur_d, cur_f_d=cur_f_d,
  #   bad_streak=bad_streak
  # )
  custom_bisection_old(
    f, cur_a, cur_b, cur_f_a, cur_f_b,
    cur_flag, abstol,
    cur_c, cur_f_c,
    cur_d, cur_f_d,
    bad_streak
  )
end#cur_c::Number=NaN, cur_f_c::Number=NaN, cur_d::Number=NaN, cur_f_d::Number=NaN, bad_streak
