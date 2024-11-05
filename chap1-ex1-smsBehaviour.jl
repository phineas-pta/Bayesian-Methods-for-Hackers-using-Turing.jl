import Statistics: mean
using Turing
using StatsPlots

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

	count_data .~ Poisson.(out)
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
