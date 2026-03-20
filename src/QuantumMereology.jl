module QuantumMereology


using PauliStrings
using LinearAlgebra

include("random.jl")
include("utils.jl")
include("spectral.jl")
include("schrieffer_wolff.jl")
include("unitary.jl")


export optimize_spectral, optimize_unitary
export projection, projection_coefficients
export GOE

end
