
using SparseArrays

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
