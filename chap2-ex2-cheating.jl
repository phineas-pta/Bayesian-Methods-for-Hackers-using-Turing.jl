"""
In the interview process for each student, the student flips a coin, hidden from the interviewer.
The student agrees to answer honestly if the coin comes up heads.
Otherwise, if the coin comes up tails, the student (secretly) flips the coin again, and answers “Yes, I did cheat” if the coin flip lands heads, and “No, I did not cheat”, if the coin flip lands tails.
This way, the interviewer does not know if a “Yes” was the result of a guilty plea, or a Heads on a second coin toss.
Thus privacy is preserved and the researchers receive honest answers.

┬ cheat = no  ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
|             |                  └ 2nd flip = heads » answer = YES
|             └ 1st flip = heads                    » answer = no
└ cheat = yes ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
              |                  └ 2nd flip = heads » answer = YES
              └ 1st flip = heads                    » answer = YES
►►► prob_yes = .5 × prob_cheat + .5² (0.5 = prob flip coin)
"""

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
