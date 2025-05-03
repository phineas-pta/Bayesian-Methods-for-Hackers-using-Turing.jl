"""
data creation algorithm:
For each data point, choose cluster 1 with probability p, else choose cluster 2.
Draw a random variate from a Normal(μᵢ, σᵢ) distribution where i was cluster chosen previously in step 1.
"""

import DelimitedFiles: readdlm
using Turing, StatsPlots
using LogExpFunctions: logaddexp

data = vec(readdlm("data/mixture_data.txt"));

# this model sucks: slow and unstable result
@model function mixture_model(data)
	N = length(data)

	# Draw locations of the components
	μ₁ ~ Normal(120, 10)
	μ₂ ~ Normal(190, 10)
	σ ~ Uniform(0, 100)

	# Draw weights
	𝒫₁ ~ Uniform(0, 1)
	𝒫₂ = 1 - 𝒫₁

	# Draw latent assignment
	z ~ filldist(Categorical([𝒫₁, 𝒫₂]), N)

	# Draw observation from selected component
	for i ∈ 1:N
		if z[i] == 1
			data[i] ~ Normal(μ₁, σ)
		else
			data[i] ~ Normal(μ₂, σ)
		end
	end
end

# faster version: marginalize discrete param
# see https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html
@model function mixture_model_bis(data)
	N = length(data)

	# Draw locations of the components
	μ₁ ~ Normal(120, 10)
	μ₂ ~ Normal(190, 10)
	σ ~ Uniform(0, 100)

	# Draw weights
	𝒫₁ ~ Uniform(0, 1) # i.e. 𝒫₂ = 1 - 𝒫₁
	for i ∈ 1:N
		cluster1 = log(    𝒫₁) + logpdf(Normal(μ₁, σ), data[i])
		cluster2 = log(1 - 𝒫₁) + logpdf(Normal(μ₂, σ), data[i])
		Turing.@addlogprob! logaddexp(cluster1, cluster2)
	end
end

chains = sample(
	#= model & data =# mixture_model_bis(data),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)

tmp = chains[["μ₁", "μ₂", "σ", "𝒫₁"]];

plot(tmp)
gelmandiag(tmp)

summarystats(tmp)
