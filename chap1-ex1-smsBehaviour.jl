"""
You are given a series of daily text-message counts from a user of your system. You are curious to know if the user’s text-messaging habits have changed over time, either gradually or suddenly. How can you model this?

text-message count of day i: Cᵢ ~ Poisson(λ)
λ = λ₁ if day < τ, λ₂ if day ≥ τ, with τ = switchpoint (user’s text-messaging habits change), if λ₁ = λ₂ then no change
λ₁ ~ Exp(α) and λ₂ ~ Exp(α)
τ ~ Unif(1, N)
"""

import Statistics: mean
using Turing, StatsPlots
using LogExpFunctions: logsumexp

const count_data = [
	13, 24,  8, 24,  7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29,  6, 19, 12, 22, 12, 18, 72, 32,  9,  7, 13,
	19, 23, 27, 20,  6, 17, 13, 10, 14,  6, 16, 15,  7,  2, 15, 15, 19, 70, 49,  7, 53, 22, 21, 31, 19, 11, 18, 20,
	12, 35, 17, 23, 17,  4,  2, 31, 30, 13, 27,  0, 39, 37,  5, 14, 13, 22,
];

@model function sms_model(count_data)
	N = length(count_data)
	α = inv(mean(count_data))

	λ₁ ~ Exponential(α)
	λ₂ ~ Exponential(α)
	τ ~ DiscreteUniform(1, N) # switchpoint

	out = zeros(N)
	out[begin:τ] .= λ₁
	out[τ:end] .= λ₂

	count_data ~ product_distribution(Poisson.(out))
	# failed to properly use `filldist` & `arraydist`
end

chains = sample(
	#= model & data =# sms_model(count_data),
	#= sampler: metropolis hastings =# MH(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)
plot(chains)
gelmandiag(chains)

mean(chains[:λ₁]), mean(chains[:λ₂]), mean(chains[:τ])
summarystats(chains)

###############################################################################
# faster version: marginalize discrete param
# see https://mc-stan.org/docs/stan-users-guide/latent-discrete.html

@model function sms_model_bis(count_data)
	λ₁ ~ Gamma(8., .3)
	λ₂ ~ Gamma(8., .3)

	N = length(count_data)

	lp₁ = fill(0., N+1)
	lp₂ = fill(0., N+1)
	for i ∈ 1:N
		lp₁[i+1] = lp₁[i] + logpdf(Poisson(λ₁), count_data[i])
		lp₂[i+1] = lp₂[i] + logpdf(Poisson(λ₂), count_data[i])
	end
	lp = (lp₂[end] - log(N)) .+ lp₁[begin:end-1] .- lp₂[begin:end-1]
	Turing.@addlogprob! logsumexp(lp)
	# cryptic error with Turing if `Turing.@addlogprob!` outside loop
end

chains_bis = sample(
	#= model & data =# sms_model_bis(count_data),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)
# cryptic error with Turing
