using Test
using LoadingMatrices

@testset "LoadingMatrices" begin
    nx = 5
    for nfactor in nx:-1:1
        nnz = ((2nx + 1) * nfactor - nfactor^2) ÷ 2
        @test nnz == nnz_loading(nx, nfactor)
        @test_throws ArgumentError nnz_loading(nx, nx+1)

        values = ones(nnz_loading(nx, nfactor))
        L = loading_matrix(values, nx, nfactor)
        @test_throws ArgumentError loading_matrix(values[2:end], nx, nfactor)
        @test_throws ArgumentError loading_matrix(values, nx+1, nfactor)
        @test size(L) == (nx, nfactor)


        k = 1
        for i in 1:nfactor
            @test all(L[i:nx, i] .== values[k:k+nx-i])
            k = k + nx-i
        end

        for i in 1:nx, j in 1:nfactor
            if j > i
                @test L[i, j] ≈ 0
            end
        end
    end
end
