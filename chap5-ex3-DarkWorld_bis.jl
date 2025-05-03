"""model with tensor manipulation instead of loop"""

import JSON: parsefile
using Turing, StatsPlots

train_data_full = parsefile("data/DarkWorld_train.json");

const XYmin = 0;
const XYmax = 4200;
const XYdims = 2;

skyID = "sky215";
n_halos = train_data_full[skyID]["n_halos"];
n_galaxies = train_data_full[skyID]["n_galaxies"];
position_galaxies = convert(
	Matrix{Float64},
	stack(train_data_full[skyID]["position_galaxies"]; dims=1) # shape: n_galaxies × XYdims
);
ellipticity_galaxies = convert(
	Matrix{Float64},
	stack(train_data_full[skyID]["ellipticity_galaxies"]; dims=1) # shape: n_galaxies × XYdims
);

const fdist_constants = [240 70 70]; # 1st large halo and 2 small ones, values independent of any sky

position_galaxies_expa = repeat(
	reshape(
		position_galaxies,
		(1, n_galaxies, XYdims)
	);
	outer=(n_halos, 1, 1)
); # shape: n_galaxies × XYdims → n_halos × n_galaxies × XYdims
fdist_cste_expa = repeat(
	reshape(
		fdist_constants[begin:n_halos],
		(n_halos, 1)
	);
	outer=(1, n_galaxies)
); # shape: n_halos → n_halos × n_galaxies

@model function halos_model_bis(n_halos, n_galaxies, position_galaxies_expa, ellipticity_galaxies)
	mass_large_halo ~ Uniform(40, 180)
	position_halos ~ filldist(Uniform(XYmin, XYmax), n_halos, XYdims) # shape: n_halos × XYdims

	mass_halos = [mass_large_halo 20 20][begin:n_halos]
	mass_halos_expa = repeat(
		reshape(
			mass_halos,
			(n_halos, 1)
		);
		outer=(1, n_galaxies)
	) # shape: n_halos → n_halos × n_galaxies
	position_halos_expa = repeat(
		reshape(
			position_halos,
			(n_halos, 1, XYdims)
		);
		outer=(1, n_galaxies, 1)
	) # shape: n_halos × XYdims → n_halos × n_galaxies × XYdims

	delta_position = position_galaxies_expa .- position_halos_expa # shape: n_halos × n_galaxies × XYdims

	matrix_euclidean_distance = sqrt.(dropdims(
		sum(
			delta_position.^2;
			dims=ndims(delta_position)
		);
		dims=ndims(delta_position)
	)) # shape: n_halos × n_galaxies × XYdims → n_halos × n_galaxies
	matrix_f_distance = max.(matrix_euclidean_distance, fdist_cste_expa) # shape: n_halos × n_galaxies

	phi_position = 2 .* atan.(delta_position[:,:,end], delta_position[:,:,begin]) # shape: n_halos × n_galaxies × XYdims → n_halos × n_galaxies
	matrix_tangential_distance = stack([-cos.(phi_position), -sin.(phi_position)]; dims=1) # shape: n_halos × n_galaxies → XYdims × n_halos × n_galaxies

	mean_ellipticity = dropdims(
		sum(
			repeat(
				reshape(
					mass_halos_expa ./ matrix_f_distance,
					(1, n_halos, n_galaxies)
				);
				outer=(XYdims, 1, 1)
			) .* matrix_tangential_distance;
			dims=2
		);
		dims=2
	)' # shape: XYdims × n_halos × n_galaxies → XYdims × n_galaxies → n_galaxies × XYdims

	for i ∈ 1:n_galaxies
		ellipticity_galaxies[i, 1] ~ Normal(mean_ellipticity[i, 1], .05)
		ellipticity_galaxies[i, 2] ~ Normal(mean_ellipticity[i, 2], .05)
	end
end

chains_bis = sample(
	#= model & data =# halos_model_bis(n_halos, n_galaxies, position_galaxies_expa, ellipticity_galaxies),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
) # may take hours

gelmandiag(chains)

summarystats(chains)
