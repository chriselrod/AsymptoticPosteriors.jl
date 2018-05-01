using RecipesBase


struct T end

# This is all we define.  It uses a familiar signature, but strips it apart
# in order to add a custom definition to the internal method `RecipesBase.apply_recipe`
@recipe function plot(ap::AsymptoticPosterior, i::Int; customcolor = :green)
    # markershape --> :auto        # if markershape is unset, make it :auto
    markercolor :=  customcolor  # force markercolor to be customcolor
    # xrotation   --> 45           # if xrotation is unset, make it 45
    # zrotation   --> 90           # if zrotation is unset, make it 90
    ( x -> pdf(ap, x, i), ap.pl.map.θ[i] - 3ap.pl.map.std_estimates[i], ap.pl.map.θ[i] + 3ap.pl.map.std_estimates[i] )                   # return the arguments (input data) for the next recipe
end