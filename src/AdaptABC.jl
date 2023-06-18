module AdaptABC

using Random
using Distances
using Distributions
using LogExpFunctions
using SpecialFunctions
using StatsBase
using LinearAlgebra
import Optim
using Manifolds
import ManifoldsBase
using Manopt
using Bijectors
using DataStructures
import ManifoldDiff
import FiniteDifferences

# probably can remove these:
import BlackBoxOptim: bboptimize, best_candidate
import LineSearches
import QuasiMonteCarlo
using Surrogates
using AbstractGPs
using SurrogatesAbstractGPs
import Zygote

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
   
    AbstractImplicitBayesianModel,
    ImplicitBayesianModel,
    TransformedImplicitBayesianModel,
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
    est, estb, logest, logestb,

    # Particle functions
    param, summ, dist,

    # Generic algorithm building blocks
    AdaptiveSMCABCState,
    adaptive_smc_generic,
    BandwidthOnlyAdaptation, ScaleReciprocal,
    OneAtATime, EuclideanLBFGS,
    ManifoldQuasiNewton, ManifoldGradientDescent,
    ManifoldNelderMead, ManifoldDifferentialEvolution,
    GlobalCovariance, LocalCovariance

# Fix broadcasting of pdf, logpdf (read: type piracy)
Broadcast.broadcastable(d::Distribution) = Ref(d)
Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{2}, f::Union{typeof(pdf), typeof(logpdf)}, d::Base.RefValue{<:MultivariateDistribution}, xs::Matrix{<:Number}) = Broadcast.broadcasted(f, d[], eachcol(xs))
Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, f::Union{typeof(pdf), typeof(logpdf)}, d::Base.RefValue{<:MultivariateDistribution}, x::Vector{<:Number}) = Broadcast.Broadcasted(f, (d, Ref(x)))
Manifolds.Sphere(::Val{n}, field::ManifoldsBase.AbstractNumbers=ℝ) where {n} = Manifolds.Sphere(n, field)
Manifolds.ProbabilitySimplex(::Val{n}) where {n} = Manifolds.ProbabilitySimplex(n)
function Random.rand!(::ProbabilitySimplex, p)
    randexp!(p)
    normalize!(p, 1)
    return nothing
end
function ManifoldsBase.get_coordinates_orthonormal!(::ProbabilitySimplex{N}, Y, p, X, R::ManifoldsBase.RealNumbers) where {N}
    ManifoldsBase.get_coordinates_orthonormal!(Sphere(Val(N)), Y, ones(N+1)/sqrt(N+1), X, R)
end
function ManifoldsBase.get_vector_orthonormal!(::ProbabilitySimplex{N}, Y, p, X, R::ManifoldsBase.RealNumbers) where {N}
    ManifoldsBase.get_vector_orthonormal!(Sphere(Val(N)), Y, ones(N+1)/sqrt(N+1), X, R)
end
function ManifoldsBase.parallel_transport_to!(P::ProbabilitySimplex, Y, p, X, q)
    project!(P, Y, q, X)
end
function ManifoldsBase.exp!(P::ProbabilitySimplex, q, p, X)
    if isapprox(X, zero_vector(P, p), rtol=1e-7)
        q .= p
    else
        s = sqrt.(p)
        Xs = X ./ s ./ 2
        θ = norm(Xs)
        q .= (cos(θ) .* s .+ Manifolds.usinc(θ) .* Xs) .^ 2
    end
    return q
end
function ManifoldsBase.log!(P::ProbabilitySimplex{N}, X, p, q) where {N}
    if isapprox(p, q, rtol=1e-7)
        fill!(X, 0)
    else
        z = sqrt.(p .* q)
        s = sum(z)
        X .= 2 * acos(s) / sqrt(1 - s^2) .* (z .- s .* p)
    end
    return X
end
function ManifoldsBase.distance(::ProbabilitySimplex, p, q)
    sumsqrt = zero(Base.promote_eltype(p, q))
    @inbounds for i in eachindex(p, q)
        sumsqrt += sqrt(p[i] * q[i])
    end
    return 2 * acos(clamp(sumsqrt, -1, 1))
end

# function ManifoldsBase.exp!(::ProbabilitySimplex, q, p, X)
#     if all(X .≈ 0)
#         q .= p
#     else
#         sp = sqrt.(p)
#         Xp = X ./ sp
#         nX = norm(Xp)
#         Xph2 = (Xp ./ nX).^2

#         q .= (p .+ Xph2)/2 .+ (p .- Xph2)/2*cos(nX) .+ Xp.*sp/nX*sin(nX)
#     end
#     return q
# end
# function ManifoldsBase.log!(P::ProbabilitySimplex{N}, X, p, q) where {N}
#     if p ≈ q
#         fill!(X, 0)
#     else
#         sp, sq = sqrt.(p), sqrt.(q)
#         dist_pq = Manifolds.distance(P, p, q)
#         dotty = dot(sp, sq)
#         X .= dist_pq/sqrt(1 - clamp(dotty, -1, 1)^2)*(sp.*sq - dotty*p)
#     end
#     return X
# end

# Not needed unless adding in a DefaultArrayStyle{N} one (e.g. always store variates along last dimension)...
#broadcasted(::DefaultArrayStyle{1}, f::typeof(pdf), d::RefValue{<:UnivariateDistribution}, xs::Vector{<:Number}) = Broadcasted(f, (d, xs))
# Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{N}, f::Union{typeof(pdf), typeof(logpdf)}, d::RefValue{<:MultivariateDistribution}, xs::Array{<:Number, N}) where {N} = broadcasted(f, d[], eachslice(xs, dims=N))
#Distributions.pdf(d::UnivariateDistribution, x::AbstractArray{<:Real, 0}) = pdf(d, first(x))
#Distributions.logpdf(d::UnivariateDistribution, x::AbstractArray{<:Real, 0}) = logpdf(d, first(x))

# Includes
include("./interface/common.jl")
include("./samplers/common.jl")

end
