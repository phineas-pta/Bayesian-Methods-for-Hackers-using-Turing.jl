import JSON: parsefile
import LinearAlgebra: diagm, I
using Turing, StatsPlots

data = parsefile("data/stocks_AAPL_GOOG_TSLA_AMZN.json");

expert_prior_μ = convert(Vector{Float64}, data["expert_mu"]);
expert_prior_σ² = diagm(data["expert_sigma"].^2);
stocks_data = data["observations"];

@model function stocks_model(stocks_data, expert_prior_μ, expert_prior_σ²)
	μ ~ MvNormal(expert_prior_μ, 10. * I)
	Σ² ~ Wishart(10., expert_prior_σ²)
	stocks_data ~ MvNormal(μ, Σ²)
end

chains = sample(
	#= model & data =# stocks_model(stocks_data, expert_prior_μ, expert_prior_σ²),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
) # VERY CRYPTIC ERROR
