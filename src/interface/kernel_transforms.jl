abstract type AbstractTransform; end
(::AbstractTransform)(args...; kwargs...) = error("Not implemented.")
pa_length(::AbstractTransform, args...; kwargs...) = error("Not implemented")

struct IdentityTransform <: AbstractTransform; end
(::IdentityTransform)(x) = x
pa_length(::IdentityTransform, ϵ::Float64) = ϵ

struct ScalingTransform{N, S} <: AbstractTransform;
    weights::Array{Float64, N}

    ScalingTransform(w::Array{Float64, N}, ::Val{S}) where {N, S} = new{N, S}(w)
    ScalingTransform(w::VecOrMat{Float64}) = ScalingTransform(w, Val(last(size(w))))
end

#! This fix is required because type-stability issues arise when using the .* for some unknown reason.
# (t::ScalingTransform{1})(x) = t.weights .* x
(t::ScalingTransform{1, S})(x) where {S} = broadcast!(*, Vector{Float64}(undef, S), t.weights, x)

(t::ScalingTransform{2})(x) = t.weights * x

Base.convert(::Type{ScalingTransform{2, S}}, t::ScalingTransform{1, S}) where {S} = ScalingTransform(diagm(t.weights), Val(S))
Base.convert(::Type{ScalingTransform{2, S}}, t::IdentityTransform) where {S} = ScalingTransform(diagm(ones(Float64, S)), Val(S))
Base.convert(::Type{ScalingTransform{1, S}}, t::IdentityTransform) where {S} = ScalingTransform(ones(Float64, S), Val(S))
pa_length(T::ScalingTransform{1}, ϵ::Float64) = ϵ ./ T.weights
# todo: Add support for this with arbitrary scaling matrices (e.g. eigs)
pa_length(T::ScalingTransform{2}, ϵ::Float64) = error("Not implemented")


