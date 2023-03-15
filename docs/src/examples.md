# Examples

## Sequential Multinomial Test

```@example sequential-multinomial-test
import Distributions
import Random
using SequentialInference

Random.seed!(0)
θ = [0.1, 0.3, 0.6]
m = Distributions.Multinomial(1, θ)
n = 1000
x = rand(m, n)
```

```@example sequential-multinomial-test
import StaticArrays

d = size(θ)[1]
θ₀ = StaticArrays.SVector{d}([0.1, 0.4, 0.5])
s = MultinomialSequence(d, θ₀ = θ₀)
update!(s, x)
dump(s)
```

```@example sequential-multinomial-test
c = ConfidenceInterval(s, 0.05)
c.interval
```

```@example sequential-multinomial-test
s = MultinomialSequence(3, θ₀ = θ₀)
ss = Array{MultinomialSequence}(undef, n)
for i in 1:n
    update!(s, x[:, [i]])
    ss[i] = deepcopy(s)
end
p = map(x -> x.pvalue, ss)
i = findfirst(p .<= 0.05)
```

```@example sequential-multinomial-test
import Cairo, Fontconfig
import DataFrames
using Gadfly

DataFrame(s::MultinomialSequence) = DataFrames.DataFrame(
    n = s.n,
    odds = exp.(s.logodds),
    pvalue = s.pvalue
)
df = vcat(DataFrame.(ss)...)
p1 = plot(df, x = :n, y = :odds, Geom.line, Scale.y_log)
p2 = plot(df, x = :n, y = :pvalue, Geom.line)
vstack(p1, p2)
```

```@example sequential-multinomial-test
DataFrame(c::ConfidenceInterval) = DataFrames.DataFrame(
    n=c.n,
    θ=1:size(c.interval)[1],
    lower=Vector(c.interval[:, 1]),
    upper=Vector(c.interval[:, 2])
)
cc = ConfidenceInterval.(ss, 0.05)
df = vcat(DataFrame.(cc)...)
plot(
    df, 
    x=:n, ymin=:lower, ymax=:upper, color=:θ,
    alpha=[0.75],
    Coord.Cartesian(ymin=0.0, ymax=1.0),
    Geom.ribbon,
    Guide.ylabel("estimate"),
    Scale.color_discrete
)
```

## Time Inhomogeneous Bernoulli Process
