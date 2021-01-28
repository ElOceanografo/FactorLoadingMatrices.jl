using Test
using FactorLoadingMatrices
using LinearAlgebra

@testset "LoadingMatrices" begin
    nx = 5
    nfactor = 3
    nnz = ((2nx + 1) * nfactor - nfactor^2) ÷ 2

    @testset "NNZ" begin
        @test nnz == nnz_loading(nx, nfactor)
        @test_throws ArgumentError nnz_loading(nx, nx+1)
    end

    @testset "Matrix" begin
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

    @testset "Varimax" begin
        values = ones(nnz_loading(nx, nfactor))
        L = loading_matrix(values, nx, nfactor)
        @test norm(varimax(L)) ≈ norm(L)
    end
end
