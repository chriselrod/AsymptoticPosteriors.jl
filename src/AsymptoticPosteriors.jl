__precompile__()

module AsymptoticPosteriors

using Compat, Optim, LineSearches, ForwardDiff, SpecialFunctions, StaticArrays, DiffEqDiffTools
using Compat.LinearAlgebra
import DiffResults
import Base.RefValue

export AsymptoticPosterior

include("false_position.jl")
include("linalg.jl")
# include("types.jl")
include("nested.jl")

end # module
