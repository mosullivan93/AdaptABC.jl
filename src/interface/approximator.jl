const SupportedKernelDistributions = Union{Uniform, Normal, Epanechnikov, Biweight, Triweight, TriangularDist, Cosine, Logistic}
# The density of the kernel family (e.g. its pdf when unscaled, i.e. bandwidth = 1)
_density(::Type{Uniform}) = Uniform(-1, 1)
_density(::Type{Normal}) = Normal()
_density(::Type{Epanechnikov}) = Epanechnikov()
_density(::Type{Biweight}) = Biweight()
_density(::Type{Triweight}) = Triweight()
_density(::Type{TriangularDist}) = TriangularDist(-1, 1)
_density(::Type{Cosine}) = Cosine()
_density(::Type{Logistic}) = Logistic()

# Inverse PDF for kernels, i.e. for what lowerbound of bandwidth will the given distance have K()/K(0) ≥ u
_inv(::Type{Uniform}, d, _) = d
_inv(::Type{Normal}, d, u) = d/sqrt(-2*log(u))
_inv(::Type{Epanechnikov}, d, u) = d/sqrt(1 - u)
_inv(::Type{Biweight}, d, u) = d/sqrt(1 - u^(1/2))
_inv(::Type{Triweight}, d, u) = d/sqrt(1 - u^(1/3))
_inv(::Type{TriangularDist}, d, u) = d/(1 - u)
_inv(::Type{Cosine}, d, u) = pi*d/acos(2u - 1)
_inv(::Type{Logistic}, d, u) = d/(2*asech(sqrt(u)))

abstract type PosteriorApproximator{D <: SupportedKernelDistributions, R <: SemiMetric, T <: AbstractTransform}; end
Broadcast.broadcastable(K::PosteriorApproximator) = Ref(K)
_density(::PosteriorApproximator{D}) where {D} = _density(D)

"A KernelRecipe contains the building blocks to construct a Kernel when given an observation as a reference and a tolerance value. In certain scenarios it can behave like a kernel, which is required when choosing an adaptive tolerance."
struct KernelRecipe{D, R, T} <: PosteriorApproximator{D, R, T}
    # todo: Check if the problem here is because it's a type variable.
    family::Type{D}
    metric::R
    transform::T

    function KernelRecipe(k::Type{D}, ρ::R, t::T) where {D <: SupportedKernelDistributions, R <: SemiMetric, T <: AbstractTransform}
        any(map(Base.Fix1(isa, ρ), Distances.weightedmetrics)) && error("Weighted metrics are restricted. Use a ScalingTransform with a weight vector to achieve the same effect.")
        isa(D, Union) && error("You must provide a single type for the distribution used as the kernel family.")
        return new{D, R, T}(k, ρ, t)
    end
end
KernelRecipe(; k::Type{<:SupportedKernelDistributions} = Uniform, ρ::SemiMetric = euclidean, t::AbstractTransform = IdentityTransform()) = KernelRecipe(k, ρ, t)
# Construct a copy of this kernel recipe with desired changes
revise(K::KernelRecipe; k=K.family, ρ=K.metric, t=K.transform) = KernelRecipe(k, ρ, t)
Base.convert(::Type{KernelRecipe{D, R, Tto}}, k::KernelRecipe{D, R, Tfrom}) where {D, R, Tfrom <: AbstractTransform, Tto <: ScalingTransform} = revise(k; t=convert(Tto, k.transform))

# Compute the distance according to the metric of the given kernel
_distance(K::KernelRecipe, T_sx, T_sy) = K.metric(T_sx, T_sy)
# Compute the distance according to the metric and transformation of the given kernel
distance(K::KernelRecipe, summ_x, summ_y) = _distance(K, K.transform(summ_x), K.transform(summ_y))
# Convenience shortcut when distance should be relative to the observation of the posterior
distance(K::KernelRecipe, mdl::ImplicitPosterior, summ_x) = distance(K, mdl.observed_summs, summ_x)

# Can be used to access the pdf of the unscaled distribution associated with the kernel's density
_pdfu(K::KernelRecipe, u::Float64) = pdf(_density(K), u)
_logpdfu(K::KernelRecipe, u::Float64) = logpdf(_density(K), u)

"Ths Kernel object represents the conditional density function of the observed vs simulated datasets."
struct Kernel{D, R, T, M} <: PosteriorApproximator{D, R, T}
    recipe::KernelRecipe{D, R, T}
    model::M
    bandwidth::Float64
    transformed_summs::Vector{Float64}

    function Kernel(m::ImplicitPosterior, ϵ::Float64, r::KernelRecipe{D, R, T}) where {D <: SupportedKernelDistributions, R <: SemiMetric, T <: AbstractTransform}
        ϵ ≥ 0 || error("Cannot have negative tolerance for posterior approximator.")
        isinf(ϵ) && @warn("Using an infinite tolerance can cause unexpected behaviour.")
        return new{D, R, T, typeof(m)}(r, m, ϵ, r.transform(m.observed_summs))
    end
end
Kernel(m::ImplicitPosterior, ϵ::Float64; k::Type{<:SupportedKernelDistributions} = Uniform, ρ::SemiMetric = euclidean, t::AbstractTransform = IdentityTransform()) = Kernel(m, ϵ, KernelRecipe(k, ρ, t))
# Construct a copy of this kernel with desired changes
revise(K::Kernel; ϵ=K.bandwidth, k=K.recipe.family, ρ=K.recipe.metric, t=K.recipe.transform) = Kernel(K.model, ϵ, KernelRecipe(k, ρ, t))
Base.convert(::Type{Kernel{D, R, Tto, M}}, k::Kernel{D, R, Tfrom, M}) where {D, R, Tfrom <: AbstractTransform, Tto <: ScalingTransform, M} = revise(k; t=convert(Tto, k.recipe.transform))

# Compute the distance of a simulation from the model with this kernel
distance(K::Kernel, summ_x) = _distance(K.recipe, K.transformed_summs, K.recipe.transform(summ_x))

# Perform the rescaling of a distance with the bandwidth and evaluate the recipe pdf.
_pdfu(K::Kernel, u::Real) = _pdfu(K.recipe, u/K.bandwidth)
_logpdfu(K::Kernel, u::Real) = _logpdfu(K.recipe, u/K.bandwidth)
# Compute the unnormalised (log)pdf for a given vector of summary statistics with this kernel
pdfu(K::Kernel, summ_x) = _pdfu(K, distance(K, summ_x))
logpdfu(K::Kernel, summ_x) = _logpdfu(K, distance(K, summ_x))
# Compute the normalised (log)pdf for a given vector of summary statistics with this kernel
Distributions.pdf(K::Kernel, summ_x) = pdfu(K, summ_x)/K.bandwidth
Distributions.logpdf(K::Kernel, summ_x) = logpdfu(K, summ_x) - log(K.bandwidth)

function pa_length(K::Kernel{D, R, T, ImplicitPosterior{M, P, S}} where {D, R, T, M, P}) where {S}
    # Using observed_summs because eventually will support transformation matrices that lower the dimension.
    pas = Vector{Float64}(undef, S)
    # The eigenvalues/eigenvectors will be from W.W' so should still be the correct shape.
    pas .= pa_length(K.recipe.transform, K.bandwidth)
    return pas
end

struct ProductKernel{D, R, T, M} <: PosteriorApproximator{D, R, T}
    kernels::Vector{Kernel{D, R, T, M}}
end
ProductKernel(ks::(Kernel{D, R, T, M})...) where {D, R, T, M} = ProductKernel(reduce(vcat, ks; init=Kernel{D,R,T,M}[]))
ProductKernel(PK::ProductKernel{D, R, T, M}, K::Kernel{D, R, T, M}) where {D, R, T, M} = ProductKernel(vcat(PK.kernels, K))
pa_length(PK::ProductKernel) = accumulate(min, transpose(hcat(pa_length.(PK.kernels)...)), dims=1)

# Default the distance on the product kernel to use the most recently added kernel
distance(K::ProductKernel, summ_x) = distance(last(K.kernels), summ_x)

# Normalised (log)pdf using product kernel can't be easily calculated
Distributions.pdf(::ProductKernel, _) = error("Exact evaluation of pdf of product kernel not possible, use unnormalised density.")
Distributions.logpdf(::ProductKernel, _) = error("Exact evaluation of logpdf of product kernel not possible, use unnormalised density.")
# Compute the unnormalised (log)pdf for a given vector of summary statistics with this kernel
pdfu(PK::ProductKernel, summ_x) = exp(logpdfu(PK, summ_x))
logpdfu(PK::ProductKernel, summ_x) = sum(logpdfu(k, summ_x) for k in PK.kernels)


#? BandwidthAdaptiveKernel
# struct BandwidthAdaptiveKernel{D <: ContinuousUnivariateDistribution} <: PosteriorApproximator{D}
# end

## ! Change this
#todo: Come back to this.
# adapt should accept a kernel recipe, a model, and samples. It also accepts an optional kwarg (ess).
# without ess, it does the same thing as the _rejection methods, with ess it targets that value.
# adapt(K::KernelRecipe; k=K.d, ρ=K.ρ, T=K.T) = KernelRecipe(k, ρ, T)
# adapt(K::Kernel{D}; ϵ=K.ϵ, ρ=K.ρ, T=K.T) where {D} = Kernel(D, ϵ, ρ, T, K.orig)