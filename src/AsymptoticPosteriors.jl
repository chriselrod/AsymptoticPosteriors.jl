__precompile__()

module AsymptoticPosteriors

using   LinearAlgebra, # special functions and LinearAlgebra ought to be clear enough.
        SpecialFunctions,
        PaddedMatrices, VectorizationBase, SIMDPirates,
        DifferentiableObjects,
        Statistics,
        ForwardDiff, DiffResults
        
# import  

using PaddedMatrices: AbstractFixedSizePaddedVector,
                    AbstractMutableFixedSizePaddedVector,
                    AbstractMutableFixedSizePaddedMatrix,
                    MutableFixedSizePaddedVector
using DifferentiableObjects: AbstractDifferentiableObject


export AsymptoticPosteriorFD, mode

# const SizedArray{P,T} = Union{SizedSIMDVector{P,T}, }

# debug() = true
# debug_rootsearch() = true
debug() = false
debug_rootsearch() = false

include("function_wrappers.jl")
include("brent.jl")
include("MAP.jl")
include("AsymptoticPosteriorInterface.jl")
include("AsymptoticPosteriorConstructors.jl")
# include("nested.jl")
include("initial_root_search.jl")
# include("plot_recipes.jl")

end # module
