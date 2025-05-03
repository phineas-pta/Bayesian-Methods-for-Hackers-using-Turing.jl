"""
On 1986/01/28, the 25th flight of the USA space shuttle program ended in disaster when one of the rocket boosters of the Shuttle Challenger exploded shortly after lift-off, killing all 7 crew members.
The presidential commission on the accident concluded that it was caused by the failure of an O-ring in a field joint on the rocket booster, and that this failure was due to a faulty design that made the O-ring unacceptably sensitive to a number of factors including outside temperature.
Of the previous 24 flights, data were available on failures of O-rings on 23, (one was lost at sea), and these data were discussed on the evening preceding the Challenger launch, but unfortunately only the data corresponding to the 7 flights on which there was a damage incident were considered important and these were thought to show no obvious trend.

observation: probability of damage incidents occurring increases as the outside temperature decreases:
probability = 1 / (1 + exp(α + β × temperature))
"""

using Turing, StatsPlots

const challenger_data = Dict(
	:temperature => [66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58],
	:damaged     => [ 0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  1]
); # use symbol instead of string to pass as argument later

@model function challenger_model(; temperature, damaged) # force keyword argument to use expanded dict
	α ~ Normal(0, 1000)
	β ~ Normal(0, 1000)
	prob = 1 ./ (1 .+ exp.(α .+ β .* temperature))
	damaged ~ product_distribution(Bernoulli.(prob))
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
