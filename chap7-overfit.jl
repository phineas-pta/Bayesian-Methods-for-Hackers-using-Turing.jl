import JSON: parsefile
using Turing, StatsPlots

data = parsefile("data/overfitting.data.json");

y = convert(Vector{Int8}, data["y"]);
X = convert(Matrix{Float64}, stack(data["X"]; dims=1));
new_X = convert(Matrix{Float64}, stack(data["new_X"]; dims=1));
# as matrix to use matrix operation later

@model function overfit_model(X, y)
	α ~ Cauchy(0, 10)
	β ~ filldist(TDist(1), size(X, 2)) # 200 var
	y .~ BernoulliLogit.(α .+ X * β)
end

chains = sample(
	#= model & data =# overfit_model(X, y),
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

predict(overfit_model(new_X, missing), chains) # VERY CRYPTIC ERROR
