import JSON: parsefile
import LinearAlgebra: diagm, I, Diagonal, Cholesky
using PDMats: PDMat
using Turing, StatsPlots

data = parsefile("data/stocks_AAPL_GOOG_TSLA_AMZN.json");

expert_prior_μ = convert(Vector{Float64}, data["expert_mu"]);
expert_prior_σ = convert(Vector{Float64}, data["expert_sigma"]); # do not raise to the 2nd power
stocks_data = convert(Matrix{Float64}, stack(data["observations"]; dims=1)'); # transpose because filldist give wrong shape

@model function stocks_model(stocks_data, expert_prior_μ, expert_prior_σ)
	μ ~ MvNormal(expert_prior_μ, 10. * I)
	Σ ~ Wishart(10., diagm(expert_prior_σ))
	stocks_data .~ filldist(MvNormal(μ, Σ), size(stocks_data, 2)) # dim 2 not 1 because transposed
end

"""more efficient Cholesky decomposition"""
@model function stocks_model_bis(stocks_data, expert_prior_μ, expert_prior_σ)
	μ ~ MvNormal(expert_prior_μ, 10. * I)
	Ω ~ LKJCholesky(length(expert_prior_σ), 2.0)
	Σ = PDMat(Cholesky(Diagonal(expert_prior_σ) * Ω.L + 1e-6 * I))
	stocks_data .~ filldist(MvNormal(μ, Σ), size(stocks_data, 2)) # dim 2 not 1 because transposed
end

chains = sample(
	#= model & data =# stocks_model_bis(stocks_data, expert_prior_μ, expert_prior_σ),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)

# write("data/chain-file.jls", chains)
# chains = read("data/chain-file.jls", Chains)

plot(chains)
gelmandiag(chains)

summarystats(chains)
