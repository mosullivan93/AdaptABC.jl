# todo: Should this with ones be the default?
struct ScalingTransform
    ws::Array{Float64}
end
_scale(w::AbstractVector, x) = w .* x
_scale(w::AbstractMatrix, x) = w * x
(T::ScalingTransform)(x) = _scale(T.ws, x)

include("./adaptive_smc_MAD.jl")
include("./adaptive_smc_opt_oaat.jl")
include("./adaptive_smc_opt_dfo.jl")
