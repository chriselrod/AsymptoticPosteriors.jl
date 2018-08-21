__precompile__()

module AsymptoticPosteriors

using   LinearAlgebra, # special functions and LinearAlgebra ought to be clear enough.
        SpecialFunctions,
        SIMDArrays,
        DifferentiableObjects,
        Statistics,
        LineSearches
# import  Optim,
        # NLSolversBase,
        # LineSearches,
import  ForwardDiff,#, StaticArrays, DiffEqDiffTools # import, so namespace access is explicit
        DiffResults

# const LinearAlgebra = Compat.LinearAlgebra

export AsymptoticPosterior, mode

# const SizedArray{P,T} = Union{SizedSIMDVector{P,T}, }

# debug() = true
# debug_rootsearch() = true
debug() = false
debug_rootsearch() = false

include("function_wrappers.jl")
# include("differentiable_objects.jl")
include("false_position.jl")
include("nested.jl")
include("initial_root_search.jl")
# include("linalg.jl")
# include("plot_recipes.jl")
# include("types.jl")

end # module
