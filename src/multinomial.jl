using Convex
using SCS
using StaticArrays

mutable struct Multinomial{d}
    d::Int64
    n::Int64
    θ₀::SVector{d, Float64}
    α₀::SVector{d, Float64}
    s::MVector{d, Int64}
    α::MVector{d, Float64}
    logodds::Float64
    pvalue::Float64
end

struct ConfidenceInterval{d}
    n::Int64
    u::Float64
    interval::SMatrix{d, 2, Float64}
end

function Multinomial(
    d::Int;
    θ₀ = ones(SVector{d, Real}) ./ d,
    k = 1.0,
    α₀ = k .* θ₀
)
    n = 0
    s = zeros(MVector{d, Int})
    α = MVector(α₀)
    logodds = 0.0
    pvalue = 1.0
    Multinomial(d, n, θ₀, α₀, s, α, logodds, pvalue)
end

function update!(s::Multinomial, x::Matrix{Int64})
    for xᵢ in eachcol(x)
        s.logodds += logbeta(s.α .+ xᵢ) - logbeta(s.α) - logpower(s.θ₀, xᵢ)
        s.pvalue = min(s.pvalue, 1 / exp(s.logodds))
        s.n += 1
        s.α .+= xᵢ
        s.s .+= xᵢ
    end
end

function ConfidenceInterval(s::Multinomial, u::Float64)
    θ = Variable(s.d)
    c = logbeta(s.α₀ + s.s) - logbeta(s.α₀)
    constraints = [c + log(u) <= sum(s.s .* log.(θ)), sum(θ) == 1]
    interval = zeros(MMatrix{s.d, 2, Float64})
    for i in 1:s.d
        lower = minimize(θ[i], constraints)
        solve!(lower, SCS.Optimizer; silent_solver = true)
        interval[i, 1] = lower.optval
        upper = maximize(θ[i], constraints)
        solve!(upper, SCS.Optimizer; silent_solver = true)
        interval[i, 2] = upper.optval
    end
    interval = SMatrix(interval)
    ConfidenceInterval(s.n, u, interval)
end
