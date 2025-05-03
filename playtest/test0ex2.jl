"""
src:
- https://www1.swarthmore.edu/NatSci/peverso1/Sports%20Data/JamesSteinData/Efron-Morris%20Baseball/EfronMorrisBB.txt
- https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
"""

using Turing, StatsPlots

const N = 18;
const at_bats = collect(Iterators.repeated(45, N));
const hits = [18, 17, 16, 15, 14, 14, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 8, 7];

@model function baseball_model(at_bats, hits)
	log_κ ~ Exponential(1.5)
	κ = exp(log_κ)
	φ ~ Uniform(0, 1)
	θ ~ Beta(φ*κ, φ*(1-κ)) # ERROR here because 1-κ is negative and julia crashed
	hits ~ product_distribution(Binomial(at_bats, θ))
end

chains = sample(
	#= model & data =# baseball_model(at_bats, hits),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)
plot(chains)
gelmandiag(chains)

summarystats(chains)
