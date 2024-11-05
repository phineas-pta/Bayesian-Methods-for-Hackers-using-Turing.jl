# Bayesian Methods for Hackers using `Turing.jl`

start julia with `--threads=auto`, check with `Threads.nthreads()`

```julia
import Pkg
Pkg.add(["Turing", "StatsPlots", "CSV", "JSON", "DataFrames"])
```

advanced read:
- http://hakank.org/julia/turing/ or https://github.com/hakank/hakank/tree/master/julia/turing
- https://storopoli.io/Bayesian-Julia/

docs:
- https://turinglang.org/docs/
- https://turinglang.org/Turing.jl/stable/api/

distributes sapling maybe faster than threaded sampling: https://discourse.julialang.org/t/trying-to-understand-parallel-performance-in-turing/74963/6
