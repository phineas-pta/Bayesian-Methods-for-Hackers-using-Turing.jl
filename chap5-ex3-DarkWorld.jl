import JSON: parsefile
using Turing, StatsPlots

train_data_full = parsefile("data/DarkWorld_train.json");

const XYmin = 0
const XYmax = 4200
const XYdims = 2

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
	mass_large_halo ~ LogUniform(40, 180)
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
		ellipticity_galaxies[i] .~ Normal.(means, .05) # shape: XYdims
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
plot(chains)
gelmandiag(chains)

summarystats(chains)
