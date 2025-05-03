"""
src:
- https://arxiv.org/pdf/1008.4686
- https://github.com/astroML/astroML/blob/main/astroML/datasets/hogg2010test.py
- https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-robust-with-outlier-detection.html
"""

using Statistics: mean
using LinearAlgebra: dot
using Turing, LogExpFunctions, StatsPlots

const N = 20;
const x    = [201., 244.,  47., 287., 203.,  58., 210., 202., 198., 158., 165., 201., 157., 131., 166., 160., 186., 125., 218., 146.];
const y    = [592., 401., 583., 402., 495., 173., 479., 504., 510., 416., 393., 442., 317., 311., 400., 337., 423., 334., 533., 344.];
const σ_x  = [  9.,   4.,  11.,   7.,   5.,   9.,   4.,   4.,  11.,   7.,   5.,   5.,   5.,   6.,   6.,   5.,   9.,   8.,   6.,   5.];
const σ_y  = [ 61.,  25.,  38.,  15.,  21.,  15.,  27.,  14.,  30.,  16.,  14.,  25.,  52.,  16.,  34.,  31.,  42.,  26.,  16.,  22.];
const ρ_xy = [-.84,  .31,  .64, -.27, -.33,  .67, -.02, -.05, -.84, -.69,   .3, -.46, -.03,   .5,  .73, -.52,   .9,   .4, -.78, -.56];

plot(x, y, xerror=σ_x, yerror=σ_y, seriestype=:scatter, label = "")

###############################################################################
# Linear Model with Custom Likelihood to Distinguish Outliers: Hogg Method
# idea: mixture model whereby datapoints can be: normal linear model vs outlier (for convenience also be linear)

@model function hogg_model(x, y, σ_y)
	b0 ~ Normal(0, 5) # weakly informative Normal priors (L2 ridge reg) for inliers
	b1 ~ Normal(0, 5)
	y_outlier ~ Normal(0, 10) # mean for all outliers
	σ_y_outlier ~ InverseGamma(.001, .001)

	# Draw weights
	𝒫_inlier ~ Uniform(0, 1) # i.e. 𝒫_outlier = 1 - 𝒫_inlier

	# faster gaussian mixture: marginalize discrete param
	for i ∈ 1:N
		inlier  = log(    𝒫_inlier) + logpdf(Normal(b0 + b1 * x[i], σ_y[i]              ), y[i])
		outlier = log(1 - 𝒫_inlier) + logpdf(Normal(y_outlier     , σ_y[i] + σ_y_outlier), y[i])
		Turing.@addlogprob! logaddexp(inlier, outlier)
	end
end

chains_hogg = sample(
	#= model & data =# hogg_model(x, y, σ_y),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)

###############################################################################
# declare outliers

N_samples, N_chains = size(chains_hogg[:b0]); # n samples thinned

# shape: N_chains × N_samples × N
log_prob_assign_outliers = [[Vector{Float64}(undef, N) for _ ∈ 1:N_samples] for _ ∈ 1:N_chains];
for i ∈ 1:N_chains, j ∈ 1:N_samples
	tmp1 = chains_hogg[:b0][j, i] .+ chains_hogg[:b1][j, i] .* x
	inliers_log_prob = logpdf.(Normal.(tmp1, σ_y), y) .+ log(chains_hogg[:𝒫_inlier][j, i])

	tmp2 = σ_y .+ chains_hogg[:σ_y_outlier][j, i]
	tmp3 = logpdf.(Normal.(chains_hogg[:y_outlier][j, i], tmp2), y)
	outliers_log_prob = tmp3 .+ log(1-chains_hogg[:𝒫_inlier][j, i])

	log_prob_assign_outliers[i][j] = outliers_log_prob .- logaddexp.(inliers_log_prob, outliers_log_prob)
end

log_prob_assign_outliers_bis = [
	[
		logsumexp(
			log_prob_assign_outliers[i][j][k]
			for j ∈ 1:N_samples
		)
		for k ∈ 1:N
	]
	for i ∈ 1:N_chains
];

proba_outlier = [exp.(log_prob_assign_outliers_bis[i] .- log(N_samples)) for i ∈ 1:N_chains];
mean_proba_outlier = [mean([proba_outlier[i][k] for i ∈ 1:N_chains]) for k ∈ 1:N];

idx_inliers = mean_proba_outlier .≤ .95
const N_inliers = sum(idx_inliers);
const x_inliers       =       x[idx_inliers];
const y_inliers       =       y[idx_inliers];
const sigma_x_inliers = sigma_x[idx_inliers];
const sigma_y_inliers = sigma_y[idx_inliers];
const rho_xy_inliers  =  rho_xy[idx_inliers];

###############################################################################
# full model with all variances

#=
const N_inlier = 17;
const x_inlier    = [201., 203.,  58., 210., 202., 198., 158., 165., 201., 157., 131., 166., 160., 186., 125., 218., 146.];
const y_inlier    = [592., 495., 173., 479., 504., 510., 416., 393., 442., 317., 311., 400., 337., 423., 334., 533., 344.];
const σ_x_inlier  = [  9.,   5.,   9.,   4.,   4.,  11.,   7.,   5.,   5.,   5.,   6.,   6.,   5.,   9.,   8.,   6.,   5.];
const σ_y_inlier  = [ 61.,  21.,  15.,  27.,  14.,  30.,  16.,  14.,  25.,  52.,  16.,  34.,  31.,  42.,  26.,  16.,  22.];
const ρ_xy_inlier = [-.84, -.33,  .67, -.02, -.05, -.84, -.69,   .3, -.46, -.03,   .5,  .73, -.52,   .9,   .4, -.78, -.56];
=#

@model function full_model(x, y, σ_x, σ_y, ρ_xy)
	# everything should be vectors of something instead of high-dimensional array
	z = [[x[i], y[i]] for i ∈ 1:N_inlier]
	cov_xy = σ_x .* σ_y .* ρ_xy
	s_bis = [[σ_x[i]^2 cov_xy[i]; cov_xy[i] σ_y[i]^2] for i ∈ 1:N_inlier]

	m ~ Normal(0, 1)
	b ~ Normal(0, 1)

	z_hat = [[x[i], b + m * x[i]] for i ∈ 1:N_inlier]
	z ~ product_distribution(MvNormal.(z_hat, s_bis))
end

chains_full = sample(
	#= model & data =# full_model(x_inlier, y_inlier, σ_x_inlier, σ_y_inlier, ρ_xy_inlier),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)

tmp = chains_full[["m", "b"]];

plot(tmp)
gelmandiag(tmp)

summarystats(tmp)

# full model with intrinsic scatter

const angle90 = π / 2;

@model function full_model_bis(x, y, σ_x, σ_y, ρ_xy)
	# everything should be vectors of something instead of high-dimensional array
	z = [[x[i], y[i]] for i ∈ 1:N_inlier]
	cov_xy = σ_x .* σ_y .* ρ_xy
	s_bis = [[σ_x[i]^2 cov_xy[i]; cov_xy[i] σ_y[i]^2] for i ∈ 1:N_inlier]

	θ ~ Uniform(-angle90, angle90) # angle of the fitted line, use this instead of slope
	v = [-sin(θ) cos(θ)]' # unit vector orthogonal to the line
	b ~ Normal(0, 1) # intercept
	V ~ InverseGamma(.001, .001) # intrinsic Gaussian variance orthogonal to the line

	for i ∈ 1:N_inlier
		delta = dot(z[i], v) - b*v[1] # orthogonal displacement of each data point from the line
		sigma2 = dot(v', s_bis[i], v) # orthogonal variance of projection of each data point to the line
		tmp = sigma2 + V # intermediary result
		Turing.@addlogprob! -.5*(log(tmp) + delta^2 / tmp) # ATTENTION sign
	end
	# cryptic error with Turing if `Turing.@addlogprob!` outside loop
end

chains_full_bis = sample(
	#= model & data =# full_model_bis(x_inlier, y_inlier, σ_x_inlier, σ_y_inlier, ρ_xy_inlier),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 5000,
	#= N chains =# 2;
	num_warmup = 1000,
	thinning = 5
)
