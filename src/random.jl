


"""
    GOE(N::Integer)

Traceless Gaussian Orthogonal Ensemble (GOE) random matrix of size `2^N x 2^N`.
"""
function GOE(N::Integer)
    dim = 2^N
    H = randn(dim, dim)
    H = H + H'
    return H-tr(H)*I/size(H, 1)
end
