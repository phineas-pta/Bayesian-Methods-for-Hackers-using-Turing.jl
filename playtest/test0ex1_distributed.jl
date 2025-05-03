using Distributed
addprocs(4);
# or start julia with `julia --procs auto`

@everywhere begin

using Turing
using LogExpFunctions: logaddexp

const N = 20
const x    = [201., 244.,  47., 287., 203.,  58., 210., 202., 198., 158., 165., 201., 157., 131., 166., 160., 186., 125., 218., 146.]
const y    = [592., 401., 583., 402., 495., 173., 479., 504., 510., 416., 393., 442., 317., 311., 400., 337., 423., 334., 533., 344.]
const σ_x  = [  9.,   4.,  11.,   7.,   5.,   9.,   4.,   4.,  11.,   7.,   5.,   5.,   5.,   6.,   6.,   5.,   9.,   8.,   6.,   5.]
const σ_y  = [ 61.,  25.,  38.,  15.,  21.,  15.,  27.,  14.,  30.,  16.,  14.,  25.,  52.,  16.,  34.,  31.,  42.,  26.,  16.,  22.]
const ρ_xy = [-.84,  .31,  .64, -.27, -.33,  .67, -.02, -.05, -.84, -.69,   .3, -.46, -.03,   .5,  .73, -.52,   .9,   .4, -.78, -.56]

end

@everywhere @model function hogg_model(x, y, σ_y)
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
	#= parallel type: threads =# MCMCDistributed(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)
