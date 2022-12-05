module AdaptABC

using Random
using Distances
using Distributions
using LogExpFunctions
using SpecialFunctions
using StatsBase
using LinearAlgebra
using Manifolds
using Manopt

export
    # The implicit model and associate algebraic tools
    ImplicitDistribution,
    IntractableLikelihood,
    IntractableTerm,
    IntractableExpression,

    prior,
    paramnames,
    summarise,
    simulate,
   
    ImplicitBayesianModel,
    ImplicitPosterior,
    PosteriorApproximator,
    Kernel,
    KernelRecipe,
    ProductKernel,

    # Samplers
    rejection_sampler,
    mcmc_sampler,
    smc_sampler,
    adaptive_smc_sampler_MAD,
    adaptive_smc_sampler_opt_oaat,
    adaptive_smc_sampler_opt_dfo,

    # Transformations
    ScalingTransform

# A few fixups for Distributions.jl
Distributions.pdf(d::UnivariateDistribution, x::AbstractArray{<:Real, 0}) = pdf(d, first(x))
Distributions.logpdf(d::UnivariateDistribution, x::AbstractArray{<:Real, 0}) = logpdf(d, first(x))
Broadcast.broadcastable(d::Distribution) = Ref(d)
dotlogpdf(d::Distribution, x::AbstractArray) = logpdf.(d, eachslice(x, dims=ndims(x))) # Not type stable until v1.9 (eachslice)
dotlogpdf(d::UnivariateDistribution, x::Real) = dotlogpdf(d, [x])

# Includes
include("./interface/common.jl")
include("./samplers/common.jl")

end
