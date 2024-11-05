using Turing, StatsPlots

const 𝒫coin = .5;
𝒩tot = 100;
𝒩yes = 35;

@model function cheating_model(𝒩tot, 𝒩yes)
	𝒫cheat ~ Uniform(0, 1)
	𝒫yes = 𝒫coin * 𝒫cheat + 𝒫coin^2 # can also use Dirac trick to make this visible in the chains
	𝒩yes ~ Binomial(𝒩tot, 𝒫yes)
	return 𝒫yes # not shown in the chains, so we have to use the `generated_quantities` function to extract that information
end

chains = sample(
	#= model & data =# cheating_model(𝒩tot, 𝒩yes),
	#= sampler: no u turn =# NUTS(),
	#= parallel type: threads =# MCMCThreads(),
	#= N samples =# 50000,
	#= N chains =# 4;
	num_warmup = 10000,
	thinning = 5
)
plot(chains)
gelmandiag(chains)

summarystats(chains)

chains_params = Turing.MCMCChains.get_sections(chains, :parameters)
genq = generated_quantities(model, chains_params)
summarystats(vcat(genq...))
