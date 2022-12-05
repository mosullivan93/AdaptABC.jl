abstract type AbstractTransform; end
(::AbstractTransform)(args...; kwargs...) = error("Not implemented.")
pa_length(::AbstractTransform, args...; kwargs...) = error("Not implemented")

struct IdentityTransform <: AbstractTransform; end
(::IdentityTransform)(x) = x
pa_length(::IdentityTransform, ϵ) = ϵ

struct ScalingTransform <: AbstractTransform
    ws::Array{Float64}
end
_scale(w::AbstractVector, x) = w .* x
_scale(w::AbstractMatrix, x) = w * x
(T::ScalingTransform)(x) = _scale(T.ws, x)
# todo: Add support for this with arbitrary scaling matrices (e.g. eigs)
pa_length(T::ScalingTransform, ϵ) = ϵ ./ T.ws
