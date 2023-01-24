module AdaptABC

using Random
using Distances
using Distributions
using LogExpFunctions
using SpecialFunctions
using StatsBase
using LinearAlgebra
using Optim
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
    adapt_ess,

    # Samplers
    rejection_sampler,
    mcmc_sampler,
    smc_sampler,
    adaptive_smc_sampler_MAD,
    adaptive_smc_sampler_opt_oaat,
    adaptive_smc_sampler_opt_dfo,

    # Transformations
    ScalingTransform,
    IdentityTransform,

    # Distance Estimation
    StatisticalDistanceEstimator,
    BhattacharyyaCoefficient,
    WeightedSampleBC,
    SubsetSampleBC,
    AdaptiveKernelEstimator,
    est, estb, logest, logestb


# Fix broadcasting of pdf, logpdf
Broadcast.broadcastable(d::Distribution) = Ref(d)
Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{2}, f::Union{typeof(pdf), typeof(logpdf)}, d::Base.RefValue{<:MultivariateDistribution}, xs::Matrix{<:Number}) = Broadcast.broadcasted(f, d[], eachcol(xs))
Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, f::Union{typeof(pdf), typeof(logpdf)}, d::Base.RefValue{<:MultivariateDistribution}, x::Vector{<:Number}) = Broadcast.Broadcasted(f, (d, Ref(x)))

# Not needed unless adding in a DefaultArrayStyle{N} one (e.g. always store variates along last dimension)...
#broadcasted(::DefaultArrayStyle{1}, f::typeof(pdf), d::RefValue{<:UnivariateDistribution}, xs::Vector{<:Number}) = Broadcasted(f, (d, xs))
# Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{N}, f::Union{typeof(pdf), typeof(logpdf)}, d::RefValue{<:MultivariateDistribution}, xs::Array{<:Number, N}) where {N} = broadcasted(f, d[], eachslice(xs, dims=N))
#Distributions.pdf(d::UnivariateDistribution, x::AbstractArray{<:Real, 0}) = pdf(d, first(x))
#Distributions.logpdf(d::UnivariateDistribution, x::AbstractArray{<:Real, 0}) = logpdf(d, first(x))

# Includes
include("./interface/common.jl")
include("./samplers/common.jl")

end
