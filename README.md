# FactorLoadingMatrices.jl
*Julia package to construct loading matrices for factor analysis*

[![Build Status](https://github.com/eloceanografo/FactorLoadingMatrices.jl/workflows/CI/badge.svg)](https://github.com/eloceanografo/FactorLoadingMatrices.jl/actions)


This is a lightweight package to construct loading matrices for probabilistic [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) and dimensionality reduction.  If you just need traditional factor analysis, that's available in [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl).  However, if you are

## Factor analysis and loading matrices
Factor analysis is a statistical method where  *n*-dimensional vector-valued variables **x** are represented as linear combination *m*-dimensional vectors of "factors" **f**. This linear combination is specified by an *n × m* loading matrix *L*,

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}_i = L \mathbf{f}_i">,

where *i* indexes each observation. If we collect all observations of **x** and **f** in the columns of matrices *X* and *F*, we can write the system as

<img src="https://render.githubusercontent.com/render/math?math={X = L F}">.

Factor analysis is useful when the elements of **x** are correlated with each other, so most of their variability can be captured using a small number of factors.  That means we can set *m < n*, and that *L* will be a rectangular matrix with more rows than columns.

There is no unique way to do this decomposition, but we want all the columns of *L* to be linearly independent.  A simple way to enforce this requirement is to set all entries above the diagonal to zero.  `FactorLoadingMatrices` exports two functions for constructing matrices with this property:

* `nnz_loading(nx, nfactor)` calculates the number of nonzero entries in a lower-triangular matrix with size `(nx, nfactor)`.
* `loading_matrix(values, nx, nfactor)` arranges the numers in the vector `values` in the lower triangle of the matrix with the specified size.

The following example shows how to use them.
```julia
julia> using FactorLoadingMatrices

julia> nx = 5;

julia> nfactor = 3;

julia> nnz = nnz_loading(5, 3)
12

julia> L = loading_matrix(randn(nnz), nx, nfactor)
5×3 Array{Float64,2}:
 -1.3836     0.0         0.0
 -1.6667     0.436374    0.0
  0.285922  -0.0585805  -1.0485
  0.512425  -0.457097   -1.43772
 -2.74867   -0.128697    0.301587
```

This package also exports a function `varimax` (modified from the implementation by Haotian Li in [NGWP.jl](https://github.com/haotian127/NGWP.jl)) to perform the [varimax](https://en.wikipedia.org/wiki/Varimax_rotation) rotation on loading matrices.

## Bayesian factor analysis using Turing

The next example shows how the utility functions in `FactorLoadingMatrices.jl` make it easy to perform Bayesian probabilistic factor analysis in Turing.

```julia
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
    # make the loading matrix
    L = loading_matrix(vals, nx, nfactor)
    # observation likelihood
    X ~ arraydist(Normal.(L * F, σ))
end

mod = FactorModel(X, nfactor)
chn = sample(mod, NUTS(), 100)
```

Once the sampler finishes running, we can extract the posterior for *L*.  Note that the matrix is sampled inside the model as a flat vector of its nonzero entries, so we have to reconstruct the loading matrices from those.

```julia
vals_post = Array(group(chn, :vals))
L_post = [loading_matrix(v, nx, nfactor) for v in eachrow(vals_post)]
```

Before comparing the matrices in `L_post` to the true `L`, we'll apply `varimax` to each one.  There's no guarantee that the model will converge on the same `L` and `F` we started with, but they should be the same (approximately) after applying a variance-maximizing rotation.

```julia
L_post_vm = varimax.(L_post)

plot(mean(L_post_vm), 1:nx, xerror=2std(L_post_vm), markerstrokecolor=1,
    layout=(1, 3), label="Model", title=["L[:, 1]" "L[:, 2]" "L[:, 3]"])
plot!(varimax(L), 1:nx, layout=(1, 3), label="Truth")
```
![MCMC posteriors for loading matrix](mcmc_factors.png)

Columns 1 and 3 of the rotated matrix were recovered quite well by the model, though it struggled a bit with column 2.

This is a relatively simple use case, but it would be straightforward to add other layers and processes to the model (e.g. covariates, non-Gaussian observations, etc.).
