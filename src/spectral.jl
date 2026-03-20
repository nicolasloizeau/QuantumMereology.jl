
using SparseArrays
using Optim
using LinearAlgebra

function buildH(h::Vector, strings::Vector{<:SparseMatrixCSC})
    H = zeros(ComplexF64, size(strings[1]))
    for (coeff, P) in zip(h, strings)
        H .+= coeff * P
    end
    return H
end


function cost_spectral(h::Vector{<:Number}, strings::Vector{<:SparseMatrixCSC}, E0::Vector{<:Number})
    H = buildH(h, strings)
    E = eigvals(Hermitian(H))
    c = real.(norm(E - E0)^2)
    return c
end


function gradient_spectral(h::Vector{<:Number}, strings::Vector{<:SparseMatrixCSC}, E0::Vector{<:Number})
    eig = eigen(Hermitian(buildH(h, strings)))
    N = round(Int, log2(size(E0, 1)))
    v = eig.vectors
    H2 = v * Diagonal(E0) * v'
    grad = zeros(promote_type(eltype(h), eltype(H2)), length(h))
    for i in eachindex(h)
        τ = strings[i]
        rows, cols, vals = findnz(τ)
        contrib = sum(H2[c, r] * val for (c, r, val) in zip(cols, rows, vals))
        grad[i] = 2 * (h[i] * 2^N - contrib)
    end
    return real.(grad)
end




"""
    optimize_spectral(H::AbstractMatrix, strings::Vector{<:SparseMatrixCSC}, iterations::Int; verbose=false)

Return a U such that U*H*U' is supported on the set of strings.
Minimize the cost function ||E - E0||^2, where E are the eigenvalues of U*H*U' and E0 are the eigenvalues of H.
https://www.pnas.org/doi/10.1073/pnas.2308006120
"""
function optimize_spectral(H::AbstractMatrix, strings::Vector{<:SparseMatrixCSC}, iterations::Int;
                    verbose=false)
    @assert ishermitian(H) "H must be Hermitian"
    E, U = eigen(Hermitian(H))
    V = optimize_spectral(E, strings, iterations; noise=noise, verbose=verbose)
    return V*U'
end


function optimize_spectral(E::Vector{<:Number}, strings::Vector{<:SparseMatrixCSC}, iterations::Int;
                    verbose=false,
                    h = (rand(length(strings)).-0.5)/ length(strings))
    cost(h) = cost_spectral(h, strings, E)
    function grad!(G,h)
        G .= gradient_spectral(h, strings, E)
    end
    function callback(state)
        verbose && println("Cost: ", state.f_x)
        return false
    end
    result = optimize(cost, grad!, h, BFGS(),
        Optim.Options(
            f_abstol = 0.0,
            x_abstol = 0.0,
            g_tol = 0.0,
            iterations = iterations,
            callback = callback
        ))
    Hp = buildH(result.minimizer, strings)
    Ep, U = eigen(Hermitian(Hp))
    return U
end
