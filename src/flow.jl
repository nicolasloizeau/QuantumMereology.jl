

function gradient_flow(H, strings, Gstrings)
    A = zeros(ComplexF64, length(strings))
    B = zeros(ComplexF64, length(Gstrings), length(strings))
    for (j, Pj) in enumerate(strings)
        A[j] = trace_product(H, Pj)
    end
    for (k, Gk) in enumerate(Gstrings)
        com = Gk*H - H*Gk
        for (j, Pj) in enumerate(strings)
            B[k, j] = trace_product(com, Pj)
        end
    end
    g = real.(2*im*B*A)
    # preconditioner
    for (k, Gk) in enumerate(Gstrings)
        g[k] /= norm(Gk*H-H*Gk) + 1e-8
    end
    return g
end


function optimize_flow(H, strings, Gstrings, iterations; verbose=false, step=0.01, max_ls=5)
    U = I
    v = zeros(length(Gstrings))
    for i in 1:iterations
        g = -gradient_flow(H, strings, Gstrings)
        g /= norm(g) + 1e-12
        # line search
        α = step
        for _ in 1:max_ls
            G = buildH(α * g, Gstrings)
            Ui = exp(-im * G)
            H_new = Ui * H * Ui'
            cost_new = -norm(projection(H_new, strings))
            cost_old = -norm(projection(H, strings))
            if cost_new < cost_old
                H = H_new
                U = Ui * U
                break  # accept step
            else
                α *= 0.5  # shrink step
            end
        end
        verbose && println("Iteration $i, cost: ", norm(H) - norm(projection(H, strings)))
    end
    return U
end
