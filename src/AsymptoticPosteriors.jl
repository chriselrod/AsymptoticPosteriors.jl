__precompile__()

module AsymptoticPosteriors

using   LinearAlgebra, # special functions and LinearAlgebra ought to be clear enough.
        SpecialFunctions,
        SIMD,
        SIMDArrays,
        jBLAS,
        DifferentiableObjects,
        Statistics
        
import  ForwardDiff,#, StaticArrays, DiffEqDiffTools # import, so namespace access is explicit
        DiffResults


export AsymptoticPosterior, mode

# const SizedArray{P,T} = Union{SizedSIMDVector{P,T}, }

# debug() = true
# debug_rootsearch() = true
debug() = false
debug_rootsearch() = false

include("function_wrappers.jl")
include("brent.jl")
include("nested.jl")
include("initial_root_search.jl")
# include("plot_recipes.jl")

end # module
