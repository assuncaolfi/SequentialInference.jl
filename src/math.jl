logbeta(v::AbstractArray) = sum(loggamma, v) - loggamma(sum(v))

# TODO: https://github.com/JuliaStats/LogExpFunctions.jl/pull/48
logpower(v::AbstractArray, w::AbstractArray) = sum(log, v .^ w)
