module LoadingMatrices

export nnz_loading, loading_matrix

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

end # module
