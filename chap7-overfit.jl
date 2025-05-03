"""
src:
- https://www.kaggle.com/c/overfitting
- http://timsalimans.com/winning-the-dont-overfit-competition/ (dead link)
- https://web.archive.org/web/20190718145349/http://timsalimans.com/winning-the-dont-overfit-competition/

In order to achieve this we have created a simulated data set with 200 variables and 20000 cases.
An ‘equation’ based on this data was created in order to generate a Target to be predicted.
Given the all 20000 cases, the problem is very easy to solve - but you only get given the Target value of 250 cases - the task is to build a model that gives the best predictions on the remaining 19750 cases.
"""

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
	y ~ product_distribution(BernoulliLogit.(α .+ X * β))
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
