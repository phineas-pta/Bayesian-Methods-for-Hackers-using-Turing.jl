"""
src:
- http://www.kaggle.com/c/DarkWorlds
- http://www.timsalimans.com/observing-dark-worlds (dead link)
- https://web.archive.org/web/20190706180949/http://timsalimans.com/observing-dark-worlds/

The dataset is actually 300 separate files, each representing a sky.
In each file, or sky, are between 300 and 720 galaxies.
Each galaxy has an x and y position associated with it, ranging from 0 to 4200, and measures of ellipticity: e1 and e2

Each sky has 1, 2 or 3 dark matter halos in it.
prior distribution of halo positions: xᵢ ~ Unif(0, 4200) and yᵢ ~ Unif(0, 4200) for i in 1,2,3

most skies had one large halo and other halos, if present, were much smaller
mass large halo ~ Unif(40, 180) | N.B. log uniform (like in original salimans solution) make MCMC struggle to find initial values
"""

import JSON: parsefile
using Turing, StatsPlots

train_data_full = parsefile("data/DarkWorld_train.json");

const XYmin = 0;
const XYmax = 4200;
const XYdims = 2;

skyID = "sky215";
n_halos = train_data_full[skyID]["n_halos"];
n_galaxies = train_data_full[skyID]["n_galaxies"];
position_galaxies = convert(Vector{Vector{Float64}}, train_data_full[skyID]["position_galaxies"]); # shape: n_galaxies × XYdims
ellipticity_galaxies = convert(Vector{Vector{Float64}}, train_data_full[skyID]["ellipticity_galaxies"]); # shape: n_galaxies × XYdims

function f_distance(position_galaxy::Vector, position_halo::Vector, cste)
	euclidean_distance = sqrt(sum((position_galaxy .- position_halo).^2))
	return max(euclidean_distance, cste)
end

const fdist_constants = [240, 70, 70]; # 1st large halo and 2 small ones

function tangential_distance(position_galaxy::Vector, position_halo::Vector)
	δ = position_galaxy .- position_halo
	ϕ = 2 * atan(δ[end], δ[begin])
	return [-cos(ϕ), -sin(ϕ)]
end

@model function halos_model(n_halos, n_galaxies, position_galaxies, ellipticity_galaxies)
	mass_large_halo ~ Uniform(40, 180)
	mass_halos = [mass_large_halo, 20, 20]
	position_halos ~ filldist(Uniform(XYmin, XYmax), n_halos, XYdims) # shape: n_halos × XYdims
	for i ∈ 1:n_galaxies
		position_galaxy = position_galaxies[i]
		tmp0 = map(1:n_halos) do j # shape: n_halos × XYdims
			position_halo = position_halos[j, :] # shape: XYdims
			tmp1 = f_distance(position_galaxy, position_halo, fdist_constants[j]) # scalar
			tmp2 = tangential_distance(position_galaxy, position_halo) # shape: XYdims
			return mass_halos[j] ./ tmp1 .* tmp2
		end # then use stack to convert vector of vectors to matrix
		means = sum(stack(tmp0; dims=1); dims=1) # shape: XYdims
		ellipticity_galaxies[i][1] ~ Normal(means[1], .05) # shape: XYdims
		ellipticity_galaxies[i][2] ~ Normal(means[2], .05) # shape: XYdims
	end
end

chains = sample(
	#= model & data =# halos_model(n_halos, n_galaxies, position_galaxies, ellipticity_galaxies),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
) # may take hours

# write("data/chain-file.jls", chains)
# chains = read("data/chain-file.jls", Chains)

plot(chains)
gelmandiag(chains)

summarystats(chains)
