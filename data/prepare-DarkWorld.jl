using CSV, DataFrames, JSON

const N = 300;

train_data_full = Dict();

# download data from https://www.kaggle.com/c/DarkWorlds/data
const data_path = "../Bayesian-Methods-for-Hackers-using-PyStan/data/DarkWorlds/";

train_data_halo_count = CSV.read(data_path * "Training_halos.csv", DataFrame, header = 1, delim = ",");

Threads.@threads for i ∈ 1:N
	n_halos = train_data_halo_count[i, "numberHalos"];
	halo_positions = [Vector(train_data_halo_count[i, ["halo_x$j", "halo_y$j"]]) for j ∈ 1:n_halos];
	train_sky = CSV.read(data_path * "Train_Skies/Training_Sky$i.csv", DataFrame, header = 1, delim = ",");
	train_data_full["sky$i"] = Dict(
		"n_halos" => n_halos,
		"halo_positions" => halo_positions,
		"n_galaxies" => nrow(train_sky),
		"position_galaxies" => Vector.(eachrow(train_sky[!, ["x", "y"]])),
		"ellipticity_galaxies" => Vector.(eachrow(train_sky[!, ["e1", "e2"]]))
	);
end

open("data/DarkWorld_train.json","w") do f
	JSON.print(f, train_data_full, 4)
end
