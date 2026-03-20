
using LinearAlgebra
using SparseArrays
using PauliStrings


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

function projection(H::AbstractMatrix, strings::Vector)
    Hp = zeros(eltype(H), size(H))
    for P in strings
        Hp .+= trace_product(H , P) * P
    end
    return Hp / size(H, 1)
end


function PauliStrings.trace_product(A::AbstractMatrix, B::AbstractMatrix)
    return tr(A*B)
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
