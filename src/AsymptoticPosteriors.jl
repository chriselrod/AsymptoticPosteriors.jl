__precompile__()

module AsymptoticPosteriors

using Compat, Optim, LineSearches, ForwardDiff, SpecialFunctions, StaticArrays, DiffEqDiffTools
import DiffResults
import Base.RefValue

export AsymptoticPosterior

include("linalg.jl")
include("types.jl")

end # module
