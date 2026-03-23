using QuantumMereology
using Test
using SparseArrays
using LinearAlgebra
using PauliStrings

@testset "spectral" begin
    N = 6
    H = GOE(N)
    H /= norm(H)
    strings = k_local_basis(N, 2)
    strings = sparse.(strings)
    iterations = 50
    U = optimize_spectral(H, strings, iterations)
    Hp = U * H * U'
    D = projection(Hp, strings)
    V = Hp - D
    @test norm(V) < 1e-6
end

# @testset "unitary" begin
#     N = 4
#     H = GOE(N)
#     H /= norm(H)
#     strings = k_local_basis(N, 2)
#     strings = sparse.(strings)
#     iterations = 50
#     U = optimize_unitary(H, strings, iterations)
#     Hp = U * H * U'
#     D = projection(Hp, strings)
#     V = Hp - D
#     @test norm(V) < 1e-6
# end


@testset "flow" begin
    N = 4
    H = GOE(N)
    H /= norm(H)
    strings = sparse.(k_local_basis(N, 2))
    Gstrings = sparse.(complete_basis(N))
    iterations = 200
    U = optimize_flow(H, strings, Gstrings, iterations)
    Hp = U * H * U'
    D = projection(Hp, strings)
    V = Hp - D
    @test norm(V) < 1e-1
end
