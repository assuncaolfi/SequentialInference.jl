module SequentialInference

using Convex
using SCS
using SpecialFunctions
using StaticArrays

export ConfidenceInterval
export DataFrame
export Multinomial
export update!

include("bernoulli.jl")
include("math.jl")
include("multinomial.jl")
include("poisson.jl")

end # module SequentialInference
