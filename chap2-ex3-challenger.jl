using Turing, StatsPlots

const challenger_data = Dict(
	:temperature => [66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58],
	:damaged     => [ 0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1]
); # use symbol instead of string to pass as argument later

@model function challenger_model(; temperature, damaged) # force keyword argument to use expanded dict
	α ~ Normal(0, 1000)
	β ~ Normal(0, 1000)
	prob = 1 ./ (1 .+ exp.(α .+ β .* temperature))
	damaged .~ Bernoulli.(prob)
end

chains = sample(
	#= model & data =# challenger_model(; challenger_data...),
	#= sampler: metropolis hastings =# MH(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)
plot(chains)
gelmandiag(chains)

summarystats(chains)
