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
