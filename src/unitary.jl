



function cost_unitary(H::AbstractMatrix, strings::Vector, U::AbstractMatrix)
    Hp = U*H*U'
    D = projection(Hp, strings)
    return -norm(D)^2
end


buildU(g, strings) = exp(im*buildH(g, strings))


function optimize_unitary(H::AbstractMatrix, strings::Vector{<:SparseMatrixCSC}, iterations::Int;
                            verbose=false,
                            Gstrings = nothing)
    @assert ishermitian(H) "H must be Hermitian"
    N = Int(log2(size(H, 1)))
    if Gstrings === nothing
        Gstrings = sparse.(complete_basis(N))
    end
    g0 = (rand(length(Gstrings)).-0.5)*2
    cost(g) = cost_unitary(H, strings, buildU(g, Gstrings))
    function callback(state)
        verbose && println("Cost: ", norm(H)+state.f_x)
        return false
    end
    result = optimize(cost, g0, LBFGS(),
        Optim.Options(
            f_abstol = 0.0,
            x_abstol = 0.0,
            g_tol = 0.0,
            iterations = iterations,
            callback = callback
        ))
    U = buildU(result.minimizer, Gstrings)
    return U
end
