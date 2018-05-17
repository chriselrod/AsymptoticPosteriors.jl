__precompile__()

module AsymptoticPosteriors

using   Compat,
        Compat.LinearAlgebra, # special functions and LinearAlgebra ought to be clear enough.
        SpecialFunctions 
import  Optim,
        NLSolversBase,
        LineSearches,
        ForwardDiff,#, StaticArrays, DiffEqDiffTools # import, so namespace access is explicit
        DiffResults

const LinearAlgebra = Compat.LinearAlgebra

export AsymptoticPosterior, mode

# debug() = true
# debug_rootsearch() = true
debug() = false
debug_rootsearch() = false

include("function_wrappers.jl")
include("differentiable_objects.jl")
include("false_position.jl")
include("nested.jl")
include("initial_root_search.jl")
include("linalg.jl")
# include("plot_recipes.jl")
# include("types.jl")

end # module
