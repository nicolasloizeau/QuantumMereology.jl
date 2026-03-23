module QuantumMereology


using PauliStrings
using LinearAlgebra

include("random.jl")
include("utils.jl")
include("spectral.jl")
include("schrieffer_wolff.jl")
include("unitary.jl")
include("flow.jl")


export optimize_spectral, optimize_unitary, optimize_flow
export projection, projection_coefficients
export GOE
export gradient

end
