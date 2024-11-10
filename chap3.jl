import DelimitedFiles: readdlm
using Turing, StatsPlots

data = vec(readdlm("data/mixture_data.txt"));

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

chains = sample(
	#= model & data =# mixture_model(data),
	#= sampler: metropolis hastings =# MH(),
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
