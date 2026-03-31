
using LinearAlgebra
using SparseArrays
using PauliStrings


function buildH(h::Vector, strings::Vector{<:SparseMatrixCSC})
    H = zeros(ComplexF64, size(strings[1]))
    for (coeff, P) in zip(h, strings)
        H .+= coeff * P
    end
    return H
end



# function buildH(h::AbstractVector, taus::Vector{<:SparseMatrixCSC})
#     @assert !isempty(taus) "taus must be non-empty"
#     @assert length(taus) == length(h) "taus and h must have same length"

#     d1, d2 = size(taus[1])
#     H = zeros(ComplexF64, d1, d2)

#     @inbounds for i in eachindex(taus)
#         A = taus[i]
#         @assert size(A) == (d1, d2) "all taus must have same shape"

#         rows = rowvals(A)
#         vals = nonzeros(A)
#         for col in 1:size(A, 2)
#             for p in nzrange(A, col)
#                 H[rows[p], col] += h[i] * vals[p]
#             end
#         end
#     end
#     return H
# end


function buildH(h::Vector, strings::Vector{<:PauliString})
    h = complex.(h)
    H = Operator(strings, h)
    set_coeffs(H, h)
    return Matrix(H)
end

function gradient(h::Vector{<:Number}, f::Function; d = 1e-6)
    grad = zeros(promote_type(eltype(h), Float64), length(h))
    dh = similar(h)
    for i in eachindex(h)
        fill!(dh, 0)
        dh[i] = d
        grad[i] = (f(h .+ dh) - f(h .- dh)) / (2d)
    end
    return real.(grad)
end


function projection_coefficients(H::AbstractMatrix, strings::Vector)
    coeffs = zeros(ComplexF64, length(strings))
    for (i, P) in enumerate(strings)
        coeffs[i] = trace_product(H, P) / size(H, 1)
    end
    return coeffs
end

function projection(H::AbstractMatrix, strings::Vector{<:SparseMatrixCSC})
    Hp = zeros(eltype(H), size(H))
    for P in strings
        Hp .+= trace_product(H , P) * P
    end
    return Hp / size(H, 1)
end

function projection(H::AbstractMatrix, strings::Vector{<:PauliString})
    Hp = zeros(eltype(H), size(H))
    for P in strings
        Hp += trace_product(H , P) * sparse(P)
    end
    return Hp / size(H, 1)
end



function PauliStrings.trace_product(A::AbstractMatrix, B::AbstractMatrix)
    return tr(A*B)
end

function PauliStrings.trace_product(A::AbstractMatrix, B::PauliString)
    return tr(A*sparse(B))
end

function PauliStrings.trace_product(A::AbstractMatrix, B::SparseMatrixCSC)
    acc = zero(promote_type(eltype(A), eltype(B)))
    for col in 1:size(B, 2)
        for ptr in B.colptr[col]:(B.colptr[col+1]-1)
            row = B.rowval[ptr]
            acc += A[col, row] * B.nzval[ptr]
        end
    end
    return acc
end
