__precompile__()

module AsymptoticPosteriors

using Compat, Optim, LineSearches, ForwardDiff, SpecialFunctions, StaticArrays
import DiffResults

export AsymptoticPosterior

include("linalg.jl")
include("types.jl")

end # module
