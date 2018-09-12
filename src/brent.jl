### Credit to djsegal
### https://discourse.julialang.org/t/is-there-a-faster-bisection-root-solver-that-uses-atol/12658/21


using Statistics
function custom_bisection(f, cur_a::Number, cur_b::Number, cur_f_a::Number, cur_f_b::Number, cur_flag::Bool=true, abstol::Number=sqrt(eps()), cur_c::Number=NaN, cur_f_c::Number=NaN, cur_d::Number=NaN, cur_f_d::Number=NaN, bad_streak::Number=0)
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

  isapprox(cur_f_a, 0.0, atol=abstol) && return cur_a
  isapprox(cur_f_b, 0.0, atol=abstol) && return cur_b
  isapprox(cur_f_c, 0.0, atol=abstol) && return cur_c
  isapprox(cur_f_d, 0.0, atol=abstol) && return cur_d

  isapprox(cur_f_a, cur_f_b, atol=abstol) && return NaN
  isapprox(cur_a, cur_b, atol=2*eps()) && return NaN

  is_bad_a = isinf(cur_f_a) || isnan(cur_f_a)
  is_bad_b = isinf(cur_f_b) || isnan(cur_f_b)
  is_bad_c = isinf(cur_f_c) || isnan(cur_f_c)
  is_bad_d = isinf(cur_f_d) || isnan(cur_f_d)

  ( !is_bad_a && !is_bad_b && cur_f_a * cur_f_b > 0 ) && return NaN

  ( is_bad_a && is_bad_b ) && return NaN

  if is_bad_a || is_bad_b || is_bad_c
    if !is_bad_d && ( isapprox(cur_c, cur_a, atol=abstol) || isapprox(cur_c, cur_b, atol=abstol) )
      cur_g = cur_d
      cur_f_g = cur_f_d
    else
      cur_g = cur_c
      cur_f_g = cur_f_c
    end

    # if is_bad_a
    #   cur_root = custom_bisection(f, cur_g, cur_b, cur_f_c, cur_f_b, abstol=abstol)
    #   isnan(cur_root) &&
    #     ( cur_root = custom_bisection(f, cur_a, cur_g, cur_f_a, cur_f_c, abstol=abstol) )
    #   return cur_root
    # else
    #   cur_root = custom_bisection(f, cur_a, cur_g, cur_f_a, cur_f_c, abstol=abstol)
    #   isnan(cur_root) &&
    #     ( cur_root = custom_bisection(f, cur_g, cur_b, cur_f_c, cur_f_b, abstol=abstol) )
    #   return cur_root
    # end
    if is_bad_a
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

  cur_d = NaN
  cur_s = zero(promote_type(typeof(cur_a),typeof(cur_b),typeof(cur_c),typeof(cur_f_a),typeof(cur_f_b),typeof(cur_f_c)))

  if !isapprox(cur_f_a, cur_f_c, atol=abstol) && !isapprox(cur_f_b, cur_f_c, atol=abstol)
    cur_term_1 = cur_f_b / ( cur_f_a - cur_f_c )
    cur_term_2 = cur_f_c / ( cur_f_a - cur_f_b )
    cur_term_3 = cur_f_a / ( cur_f_b - cur_f_c )

    cur_s += cur_a * cur_term_1 * cur_term_2
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

  cur_bool = ( cur_s < cur_b && cur_s < ( ( 3 * cur_a + cur_b ) / 4 ) )
  cur_bool |= ( cur_s > cur_b && cur_s > ( ( 3 * cur_a + cur_b ) / 4 ) )

  if cur_flag
    cur_diff = abs( cur_b - cur_c )
  else
    cur_diff = abs( cur_d - cur_c )
  end

  cur_bool |= abs( cur_s - cur_b ) >= ( cur_diff / 2 )
  cur_bool |= isapprox(cur_diff, 0.0, atol=abstol)

  cur_flag = cur_bool
  cur_bool && ( cur_s = middle(cur_a, cur_b) )

  cur_d = cur_c
  cur_c = cur_b

  cur_f_d = cur_f_c
  cur_f_c = cur_f_b

  cur_f_s = f(cur_s)

  if cur_f_a * cur_f_s < 0
    cur_b = cur_s
    cur_f_b = cur_f_s
  else
    cur_a = cur_s
    cur_f_a = cur_f_s
  end

  is_bad_f_a = isapprox(cur_f_a, init_f_a, atol=abstol)
  is_bad_f_b = isapprox(cur_f_b, init_f_b, atol=abstol)

  is_bad_f_a |= abs(cur_f_a) > abs(init_f_a)
  is_bad_f_b |= abs(cur_f_b) > abs(init_f_b)

  if ( is_bad_f_a && is_bad_f_b )
    bad_streak += 1
  else
    bad_streak = 0
  end

  ( bad_streak > 4 ) && return NaN

  is_unchanged = isapprox(cur_a, init_a, atol=2*eps())
  is_unchanged &= isapprox(cur_b, init_b, atol=2*eps())

  if is_unchanged
    cur_c = middle(cur_a, cur_b)
    cur_f_c = f(cur_c)

    # cur_root = custom_bisection(f, cur_a, cur_c, cur_f_a, cur_f_c, abstol=abstol, bad_streak=bad_streak)
    cur_root = custom_bisection(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol, NaN, NaN, NaN, NaN, bad_streak)
    isnan(cur_root) || return cur_root

    # cur_root = custom_bisection(f, cur_c, cur_b, cur_f_c, cur_f_b, abstol=abstol, bad_streak=bad_streak)
    # return cur_root
    return custom_bisection(f, cur_a, cur_b, cur_f_a, cur_f_b, cur_flag, abstol, NaN, NaN, NaN, NaN, bad_streak)
  end

  # custom_bisection(
  #   f, cur_a, cur_b, cur_f_a, cur_f_b,
  #   cur_flag, abstol=abstol,
  #   cur_c=cur_c, cur_f_c=cur_f_c,
  #   cur_d=cur_d, cur_f_d=cur_f_d,
  #   bad_streak=bad_streak
  # )
  custom_bisection(
    f, cur_a, cur_b, cur_f_a, cur_f_b,
    cur_flag, abstol,
    cur_c, cur_f_c,
    cur_d, cur_f_d,
    bad_streak
  )
end#cur_c::Number=NaN, cur_f_c::Number=NaN, cur_d::Number=NaN, cur_f_d::Number=NaN, bad_streak
