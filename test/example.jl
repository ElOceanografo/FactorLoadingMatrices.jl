using FactorLoadingMatrices
using Random
using Plots
using Turing
# using Memoization and reverse-mode autodiff speeds up MCMC sampling
# *a lot* when fitting high-dimensional models
using Memoization
using ReverseDiff
Turing.setadbackend(:reversediff)

Random.seed!(1)

nx = 50
nfactor = 3
nobs = 300
σ = 0.5

vals = randn(nnz_loading(nx, nfactor))
L = loading_matrix(vals, nx, nfactor)

F = randn(nfactor, nobs)
X = L * F .+ σ*randn()

@model function FactorModel(X, nfactor)
    nx, nobs = size(X)
    # set priors for factors, loading matrix, and observation noise
    F ~ filldist(Normal(0, 1), nfactor, nobs)
    vals ~ filldist(Normal(0, 2), nnz_loading(nx, nfactor))
    σ ~ Exponential(1.0)

    L = loading_matrix(vals, nx, nfactor)
    X ~ arraydist(Normal.(L * F, σ))
end

mod = FactorModel(X, nfactor)
chn = sample(mod, NUTS(), 100)

vals_post = Array(group(chn, :vals))
L_post = [loading_matrix(v, nx, 3) for v in eachrow(vals_post)]

L_post_vm = varimax.(L_post)
plot(mean(L_post_vm), 1:nx, xerror=2std(L_post_vm), markerstrokecolor=1,
    layout=(1, 3), label="Model", title=["L[:, 1]" "L[:, 2]" "L[:, 3]"])
plot!(varimax(L), 1:nx, layout=(1, 3), label="Truth")

savefig("mcmc_factors.png")
