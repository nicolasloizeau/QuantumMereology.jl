



function cost_unitary(H::AbstractMatrix, strings::Vector, U::AbstractMatrix)
    Hp = U*H*U'
    D = projection(Hp, strings)
    return -norm(D)^2
end


function cost_unitary2(H::AbstractMatrix, strings::Vector, G::AbstractMatrix)
    com1 = G*H - H*G
    com2 = G*com1 - com1*G
    Hp = H + im*com1
    D = projection(Hp, strings)
    return -norm(D)^2
end




function buildA(H, Gstrings, strings)
    A = zeros(ComplexF64, length(Gstrings), length(strings))
    for (k, Gk) in enumerate(Gstrings)
        com = Gk*H - H*Gk
        for (j, Pj) in enumerate(strings)
            A[k, j] = trace_product(com, Pj)
        end
    end
    return A
end


function optimize_unitary(H::AbstractMatrix, strings::Vector{<:SparseMatrixCSC}, iterations::Int;
                            verbose=false,
                            Gstrings = nothing)
    @assert ishermitian(H) "H must be Hermitian"
    N = Int(log2(size(H, 1)))
    if Gstrings === nothing
        Gstrings = sparse.(complete_basis(N))
    end
    g0 = (rand(length(Gstrings)).-0.5)*2
    buildU(g, strings) = exp(im*buildH(g, strings))
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
