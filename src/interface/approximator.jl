abstract type PosteriorApproximator{D} end
Broadcast.broadcastable(K::PosteriorApproximator) = Ref(K)

"A KernelRecipe contains the building blocks to construct a Kernel when given an observation as a reference and a tolerance value. In certain scenarios it can behave like a kernel, which is required when choosing an adaptive tolerance."
struct KernelRecipe{D <: ContinuousUnivariateDistribution, R <: SemiMetric, F} <: PosteriorApproximator{D}
    d::Type{D}
    ρ::R
    T::F
end
_kernel(::KernelRecipe{D}) where {D} = _kernel(D)
distance(K::KernelRecipe, x, y) = K.ρ(K.T(x), K.T(y))
distance(K::KernelRecipe, mdl::ImplicitPosterior, x::Vector{Float64}) = K.ρ(K.T(x), K.T(mdl.observed_summs))
KernelRecipe(;k::Type{<:ContinuousUnivariateDistribution} = Uniform, ρ::SemiMetric = euclidean, T = identity) = KernelRecipe(k, ρ, T)
adapt(K::KernelRecipe; k=K.d, ρ=K.ρ, T=K.T) = KernelRecipe(k, ρ, T)

"Ths Kernel object represents the conditional density function of the observed vs simulated datasets."
struct Kernel{D <: ContinuousUnivariateDistribution, E <: Number, R <: SemiMetric, F, O} <: PosteriorApproximator{D}
    d::Type{D}
    ϵ::E
    ρ::R
    T::F
    obs::O
    orig

    function Kernel(d::Type{D}, e::E, r::R, t::F, o) where {D, E, R, F}
        e ≥ 0 || error("Cannot have negative tolerance for posterior approximator.")
        isinf(e) && @warn("Using an infinite tolerance can cause unexpected behaviour.")
        local oo = t(o)
        return new{D, E, R, F, typeof(oo)}(d, e, r, t, oo, o)
    end
end
# Allow us to create a recipe that matches this kernel
KernelRecipe(K::Kernel{D}; k=K.d, ρ=K.ρ, T=K.T) where {D} = KernelRecipe(k, ρ, T)
# Default the construction to a uniform.
Kernel(summs, ϵ; k::Type{<:ContinuousUnivariateDistribution} = Uniform, ρ::SemiMetric = euclidean, T = identity) = Kernel(k, float(ϵ), ρ, T, summs)
Kernel(m::ImplicitPosterior, ϵ; k::Type{<:ContinuousUnivariateDistribution} = Uniform, ρ::SemiMetric = euclidean, T = identity) = Kernel(k, float(ϵ), ρ, T, m.observed_summs)
Kernel(summs, ϵ, K::KernelRecipe) = Kernel(K.d, float(ϵ), K.ρ, K.T, summs)
Kernel(m::ImplicitPosterior, ϵ, K::KernelRecipe) = Kernel(m.observed_summs, ϵ, K)
adapt(K::Kernel{D}; ϵ=K.ϵ, ρ=K.ρ, T=K.T) where {D} = Kernel(D, ϵ, ρ, T, K.orig)
# use getindex for subset selection

_kernel(::Type{Uniform}) = Uniform(-1, 1)
_kernel(::Type{Normal}) = Normal()
_kernel(::Type{Epanechnikov}) = Epanechnikov()
_kernel(::Type{Biweight}) = Biweight()
_kernel(::Type{Triweight}) = Triweight()
_kernel(::Type{TriangularDist}) = TriangularDist(-1, 1)
_kernel(::Type{Cosine}) = Cosine()
_kernel(::Type{Logistic}) = Logistic()
_kernel(::Kernel{D}) where {D} = _kernel(D)

distance(K::Kernel, x) = K.ρ(K.T(x), K.obs)

(K::Kernel)(x) = K(x, Val(:pdf))
(K::Kernel)(x, ::Val{:pdf}) = pdf(_kernel(K), distance(K, x)/K.ϵ)/K.ϵ
(K::Kernel)(x, ::Val{:logpdf}) = logpdf(_kernel(K), distance(K, x)/K.ϵ) - log(K.ϵ)
(K::Kernel)(x, ::Val{:pdf_prop}) = pdf(_kernel(K), distance(K, x)/K.ϵ)
(K::Kernel)(x, ::Val{:logpdf_prop}) = logpdf(_kernel(K), distance(K, x)/K.ϵ)

# Inverse PDF for kernels, i.e. for what lowerbound of bandwidth will the given distance have K()/K(0) ≥ u
_inv(::Type{Uniform}, d, _) = d
_inv(::Type{Normal}, d, u) = d/sqrt(-2*log(u))
_inv(::Type{Epanechnikov}, d, u) = d/sqrt(1 - u)
_inv(::Type{Biweight}, d, u) = d/sqrt(1 - u^(1/2))
_inv(::Type{Triweight}, d, u) = d/sqrt(1 - u^(1/3))
_inv(::Type{TriangularDist}, d, u) = d/(1 - u)
_inv(::Type{Cosine}, d, u) = pi*d/acos(2u - 1)
_inv(::Type{Logistic}, d, u) = d/(2*asech(sqrt(u)))

struct ProductKernel{D <: ContinuousUnivariateDistribution} <: PosteriorApproximator{D}
    ks::Vector{<: Kernel{D}}
end
ProductKernel(Ks::Kernel{D}...) where {D} = ProductKernel([Ks...])
ProductKernel(PK::ProductKernel{D}, K::Kernel{D}) where {D} = ProductKernel([PK.ks..., K])

# Default to the most recently appended kernel.
distance(K::ProductKernel, x) = distance(last(K.ks), x)

(K::ProductKernel)(x) = exp(K(x, Val(:logpdf)))
(K::ProductKernel)(x, ::Val{:logpdf}) = sum(k(x, Val(:logpdf)) for k in K.ks)
(K::ProductKernel)(x, ::Val{:pdf_prop}) = exp(K(x, Val(:logpdf_prop)))
(K::ProductKernel)(x, ::Val{:logpdf_prop}) = sum(k(x, Val(:logpdf_prop)) for k in K.ks)
