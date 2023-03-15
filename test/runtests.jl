using Revise

import DataFrames
import Distributions
import Random
using SequentialInference
using StaticArrays
using Test

@testset "Sequential Multinomial Test" begin

    Random.seed!(0)
    θ = [0.1, 0.3, 0.6]
    m = Distributions.Multinomial(1, θ)
    n = 1000
    x = rand(m, n)
    @test x[:, 1] == [0, 1, 0]
    @test x[:, n] == [0, 0, 1]

    d = size(θ)[1]
    θ₀ = SVector{d}([0.1, 0.4, 0.5])
    s = MultinomialSequence(d, θ₀ = θ₀)
    update!(s, x)
    @test s.d == d
    @test s.n == n
    @test s.θ₀ == θ₀
    @test s.α₀ == [0.1, 0.4, 0.5]
    @test s.s == [122, 303, 575]
    @test s.α == [122.1, 303.4, 575.5]
    @test s.logodds == 12.739503195055113
    @test s.pvalue == 1.9254538401356433e-6

    c = ConfidenceInterval(s, 0.05)
    @test c.interval == [
        0.07948503786187615 0.175202564144606;
        0.23877864561274295 0.3727855255640303;
        0.5018595841193678 0.64595318571502
    ]

    s = MultinomialSequence(3, θ₀ = θ₀)
    ss = Array{MultinomialSequence}(undef, n)
    for i in 1:n
        update!(s, x[:, [i]])
        ss[i] = deepcopy(s)
    end
    p = map(x -> x.pvalue, ss)
    i = findfirst(p .<= 0.05)
    @test i == 162

end


@testset "Time Inhomogeneous Bernoulli Process" begin

    ρ = [0.1, 0.3, 0.6]
    # g(i) =
    δ = [log(0.2), log(0.3), log(0.4)]
    μ(i) = 1/2 * sin(7πi/n) + 1/2
    f(i) = exp(μ(i)) * exp()

end
