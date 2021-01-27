module LoadingMatrices
"""
Gives the number of entries in the lower triangle of a matrix with the specified number of
rows and columns.
"""
lower_triangle_dimension(nrow, ncol) = ((2nrow + 1) * ncol - ncol^2) ÷ 2

"""
Constructs a matrix of loadings to map `nfactor` latent variables to `ndata` observed
variables.  The upper triangle is all zeros to enforce linear independence among the loading
vectors.
# Arguments
- `values::AbstractVector`: vector of values to put in the nonzero lower triangle. They are
filled in order running down the columns from left to right.
- `ndata::Integer`: Dimension of the data, i.e. the number of rows in the loading matrix
- `nfactor::Integer`: Number of factors, i.e. the number of columns in the loading matrix
"""
function loading_matrix(values::AbstractVector, ndata::Integer, nfactor::Integer)
    if length(values) != lower_triangle_dimension(ndata, nfactor)
        DimensionMismatch("Wrong number of data values for a matrix of the specified size.")
    end
    L = zeros(eltype(values), ndata, nfactor)
    k = 1
    for i in 1:ndata, j in 1:nfactor
        if i >= j
            L[i, j] = i >= j ? values[k] : 0
            k += 1
        end
    end
    return L
end

end # module
