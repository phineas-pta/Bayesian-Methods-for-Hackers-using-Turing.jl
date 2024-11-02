import Random: seed!
import Statistics: mean
using Turing
using StatsPlots
seed!(123)

# those values are unknown
true𝒫A = .05;
true𝒫B = .04;

# sample size
𝒩A = 1500;
𝒩B = 750;

# generate data
dataA = rand(Bernoulli(true𝒫A), 𝒩A);
dataB = rand(Bernoulli(true𝒫B), 𝒩B);

@model function ABtesting_model(dataA, dataB)
	𝒫A ~ Uniform(0, 1)
	𝒫B ~ Uniform(0, 1)
	dataA .~ Bernoulli(𝒫A)
	dataB .~ Bernoulli(𝒫B)
	Δ𝒫 ~ Dirac(𝒫A - 𝒫B) # Dirac trick to make this visible in the chains
	# return 𝒫A - 𝒫B # another way to extract this info: use `generated_quantities` function
end

chains = sample(
	#= model & data =# ABtesting_model(dataA, dataB),
	#= sampler: metropolis hasting =# MH(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)
plot(chains)
gelmandiag(chains)

mean(chains[:Δ𝒫])
summaries, _ = describe(chains);
