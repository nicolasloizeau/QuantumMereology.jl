
function commutator(A, B)
    return A * B - B * A
end


function gradient_flow(H, strings, Gstrings)
    if typeof(strings[1]) <: PauliString
        H = Operator(H)
    end
    A = zeros(ComplexF64, length(strings))
    B = zeros(ComplexF64, length(Gstrings), length(strings))
    for (j, Pj) in enumerate(strings)
        A[j] = trace_product(H, Pj)
    end
    for (k, Gk) in enumerate(Gstrings)
        com = commutator(Gk, H)
        for (j, Pj) in enumerate(strings)
            B[k, j] = trace_product(com, Pj)
        end
    end
    g = real.(2*im*B*A)
    # preconditioner
    for (k, Gk) in enumerate(Gstrings)
        g[k] /= norm(commutator(Gk, H)) + 1e-8
    end
    return g
end



function line_search(H, g, strings, step, buildU)
    Ui = I
    for _ in 1:5
        Ui = buildU(step * g)
        H_new = Ui * H * Ui'
        if norm(projection(H_new, strings)) > norm(projection(H, strings))
            H = H_new
            break  # accept step
        else
            step *= 0.5  # shrink step
        end
    end
    return H, Ui, step
end

"""
buildU is a functiont that takes a vector of coefficients and return the unitary step.
"""
function optimize_flow(H::AbstractMatrix, strings, Gstrings, iterations; verbose=false, step=0.01, buildU=nothing, noise=1e-3)
    U = I
    g = rand(length(Gstrings)).-0.5
    if buildU === nothing
        buildU = g -> exp(-im * buildH(g, Gstrings))
    end
    for i in 1:iterations
        g = -gradient_flow(H, strings, Gstrings)
        g /= norm(g) + 1e-12
        g += (rand(length(g)).-0.5) * noise
        H, Ui, step = line_search(H, g, strings, step, buildU)
        step *= 1.2
        U = Ui * U
        verbose && println("Iteration $i, cost: ", norm(H) - norm(projection(H, strings)))
    end
    return U
end
