module FactorLoadingMatrices
using LinearAlgebra

export nnz_loading, loading_matrix, varimax

"""
Gives the number of nonzero entries (i.e. the number of entries in the lower
triangle) of the loading matrix for the specified data dimension `nx` and number
of factors `nfactor`.
"""
function nnz_loading(nx, nfactor)
    nx >= nfactor || throw(ArgumentError("`nfactor` ($nfactor) must be <= `nx` ($nx)"))
    return ((2nx + 1) * nfactor - nfactor^2) ÷ 2
end


"""
Constructs a matrix of loadings to map `nfactor` latent variables to `nx` observed
variables.  The upper triangle is all zeros to enfactororce linear independence among the loading
vectors.
# Arguments
- `values::AbstractVector`: vector of values to put in the nonzero lower triangle. They are
filled in order running down the columns from left to right.
- `nx::Integer`: Dimension of the data, i.e. the number of rows in the loading matrix
- `nfactor::Integer`: Number of factors, i.e. the number of columns in the loading matrix
"""
function loading_matrix(values::AbstractVector, nx::Integer, nfactor::Integer)
    if length(values) != nnz_loading(nx, nfactor)
        throw(ArgumentError("Wrong number of data values for a loading matrix of the specified size."))
    end
    L = zeros(eltype(values), nx, nfactor)
    k = 1
    for i in 1:nx, j in 1:nfactor
        if i >= j
            L[i, j] = i >= j ? values[k] : 0
            k += 1
        end
    end
    return L
end

"""
	varimax(A; gamma = 1.0, minit = 20, maxit = 1000, reltol = 1e-12)
VARIMAX perform varimax (or quartimax, equamax, parsimax) rotation to the column vectors of the input matrix.
# Input Arguments
- `A::Matrix{TA}`: input matrix, whose column vectors are to be rotated. d, m = size(A).
- `gamma`: default is 1. gamma = 0, 1, m/2, and d(m - 1)/(d + m - 2), corresponding to quartimax, varimax, equamax, and parsimax.
- `minit::Int`: default is 20. Minimum number of iterations, in case of the stopping criteria fails initially.
- `maxit::Int`: default is 1000. Maximum number of iterations.
- `reltol::Float64`: default is 1e-12. Relative tolerance for stopping criteria.
# Output Argument
- `B::Matrix{Float64}`: output matrix, whose columns are already been rotated.
Implemented by Haotian Li, Aug. 20, 2019
"""
function varimax(A; gamma = 1.0, minit = 20, maxit = 1000, reltol = 1e-12)
	d, m = size(A)
	m == 1 && return A
	TA = eltype(A)

	# Warm up step: start with a good initial orthogonal matrix T by SVD and QR
	T = Matrix{TA}(I, m, m)
	B = A * T
	L,_,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
	T = L * M'
	if norm(T-Matrix{TA}(I, m, m)) < reltol
		T,_ = qr(randn(m,m)).Q
		B = A * T
	end

	# Iteration step: get better T to maximize the objective (as described in Factor Analysis book)
	D = 0
	for k in 1:maxit
		Dold = D
		L,s,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
		T = L * M'
		D = sum(s)
		B = A * T
		if (abs(D - Dold)/D < reltol) && k >= minit
			break
		end
	end

	# Adjust the sign of each rotated vector such that the maximum absolute value is positive.
	for i in 1:m
		if abs(maximum(B[:,i])) < abs(minimum(B[:,i]))
			B[:,i] .= - B[:,i]
		end
	end

	return B
end

end # module
