# abstract type EstimatorInputForm; end
# abstract type MaskedSamples <: EstimatorInputForm; end
# abstract type WeightedSamples <: EstimatorInputForm; end
# abstract type LogWeights <: WeightedSamples; end
# abstract type LinearWeights <: WeightedSamples; end

# abstract type StatisticalDistanceEstimator{W<:EstimatorInputForm}; end
# # evaluate(e::StatisticalDistanceEstimator, mask_or_weights; output_scale::Union{Val{:LogScale}, Val{:LinearScale}}=Val(:LogScale), normalised::Union{Val{true}, Val{false}}=Val(false)) = eva

# abstract type AlphaDivergence{α, W} <: StatisticalDistanceEstimator{W}; end
# # Weighted and Masked Estimators

# est(...)
# estu(...)
# logest(...)
# logestu(...)


abstract type StatisticalDistanceEstimator; end
(e::StatisticalDistanceEstimator)(args...; kwargs...) = exp(e(Val(:log), args...; kwargs...))
(::StatisticalDistanceEstimator)(::Val{:log}, args...; kwargs...) = error("Not implemented")

# todo: Make the Renyi the default? Have alpha as a type parameter?

abstract type BhattacharyyaCoefficient <: StatisticalDistanceEstimator; end
abstract type ProportionalBhattacharyyaCoefficient <: BhattacharyyaCoefficient; end
_logbias(::ProportionalBhattacharyyaCoefficient, args...; kwargs...) = 0
logbias(bc::ProportionalBhattacharyyaCoefficient, args...; kwargs...) = loggamma(bc.k) - loggamma(bc.k + 1/2) + (bc.d/2*log(π) - loggamma(bc.d/2 + 1))/2 + _logbias(bc, args...; kwargs...)
struct ExactBhattacharyyaCoefficient{T<:ProportionalBhattacharyyaCoefficient} <: BhattacharyyaCoefficient
    int_bc::T

    ExactBhattacharyyaCoefficient{E}(ps::AbstractArray, q::ContinuousDistribution; k::Union{Missing,Int}=missing) where {E<:ProportionalBhattacharyyaCoefficient} = new(E(ps, q; k=k))
end
(bc::ExactBhattacharyyaCoefficient)(v::Val{:log}, args...; kwargs...) = bc.int_bc(v, args...; kwargs...) + logbias(bc.int_bc, args...; kwargs...)

# todo: test keep whole sorted matrix to compare all k's efficiently (exploratory purposes).
# todo: ^^ might be required to facilitate allow k to vary per point (may help with the weighted version?).
# note: At what point do I just use a gaussian kde?
# todo: examine the analytical expectation of the weighted distance estimator when weights are uniform.
# todo: encapsulate with a helper type that accepts a kernel recipe and evaluates the BC for a particular transform
# todo: ^^^ automatic rescaling of the summary statistics

silverman_rule_of_thumb(ps::AbstractVector) = 0.9*min(std(ps; corrected=false), iqr(ps)/1.34)*(length(ps))^(-1/5)
function silverman_rule_of_thumb(ps::AbstractMatrix)
    d, N = size(ps)
    return (4/(d+2))^(1/(d+4))*sqrt(mean(var(ps, dims=2)))*N^(-1/(d+4))
end
adaptive_k(ps, nus) = sum(median(nus, dims=2) .< silverman_rule_of_thumb(ps)) + 1

struct WeightedSampleBC <: ProportionalBhattacharyyaCoefficient
    # Log of the non-weight terms in the summand of the estimator.
    ts::Vector{Float64}
    # Number of samples
    N::Int
    # Dimension of the problem samples
    d::Int
    # Technically the k could be adapted after applying a mask... here it is done beforehand because the bias depends on it.
    k::Int

    function WeightedSampleBC(ps::AbstractArray, q::ContinuousDistribution; k::Union{Missing,Int}=missing)
        sz = collect(size(ps))
        N = pop!(sz)
        d = prod(sz) # Not sure if prod would make sense here (e.g. matrix variates), but conveniently returns 1 for empty collections.

        nus = zeros(Float64, (N, N))
        pairwise!(nus, euclidean, ps)
        nus[iszero.(nus)] .= Inf
        ndups = count(isinf, nus, dims=2)
        sort!(nus, dims=1)

        lλs = log.(N .- ndups)
        lqs = logpdf.(q, ps)

        if ismissing(k)
            k = adaptive_k(ps, nus)
        end
        lηs = log.(nus[k, :])

        return new(vec(lqs .+ lλs .+ d*lηs), N, d, k)
    end
end
_logbias(bc::WeightedSampleBC, _) = -log(bc.N)/2
(bc::WeightedSampleBC)(::Val{:log}, lws) = logsumexp((lws .+ bc.ts)/2) - logsumexp(lws)/2

struct SubsetSampleBC <: ProportionalBhattacharyyaCoefficient
    # Pairwise distances in the full sample.
    nus::Matrix{Float64}
    # logpdf of the reference q at each point in the full sample
    lqs::Vector{Float64}
    # Number of samples
    N::Int
    # Dimension of the problem samples
    d::Int
    # Technically the k could be adapted after applying a mask... here it is done beforehand because the bias depends on it.
    k::Int

    function SubsetSampleBC(ps::AbstractArray, q::ContinuousDistribution; k::Union{Missing,Int}=missing)
        sz = collect(size(ps))
        N = pop!(sz)
        d = prod(sz) # Not sure if prod would make sense here

        nus = zeros(Float64, (N, N))
        pairwise!(nus, euclidean, ps)
        nus[iszero.(nus)] .= Inf
        
        lqs = logpdf.(q, ps)

        if ismissing(k)
            # adapting k will cost at least one sort operation.
            k = adaptive_k(ps, sort(nus, dims=1))
        end

        return new(nus, lqs, N, d, k)
    end
end
_logbias(bc::SubsetSampleBC, idx) = -log(length(first(to_indices(bc.lqs, (idx,)))))
function (bc::SubsetSampleBC)(::Val{:log}, idx)
    kept_nus = bc.nus[idx, idx]
    ndups = count(isinf, kept_nus, dims=2)
    lλs = log.(size(kept_nus, 2) .- ndups)

    sort!(kept_nus, dims=1, alg=PartialQuickSort(bc.k))
    lηs = log.(kept_nus[bc.k, :])

    return logsumexp((bc.lqs[idx] .+ lλs .+ bc.d*lηs)/2)
end

# struct BootstrappedExactBhattacharyyaCoefficient{T<:ProportionalBhattacharyyaCoefficient} <: BhattacharyyaCoefficient
#     # Vector of estimators
#     ests::Vector{T}
#     # Vector of the bootstrapped indices
#     idxs::Vector{Vector{Int}}

#     function BootstrappedExactBhattacharyyaCoefficient{E}(
#         ps::AbstractArray, q::ContinuousDistribution;
#         bs::Int=10, nbs=missing, k::Union{Missing,Int}=missing) where {E <: ProportionalBhattacharyyaCoefficient}

#         sz = collect(size(ps))
#         N = pop!(sz)
#         nbs = ismissing(nbs) ? N : nbs

#         if ismissing(k)
#             # adapting k will cost a distance and sort operation on the original set.
#             nus = zeros(Float64, (N, N))
#             pairwise!(nus, euclidean, ps)
#             nus[iszero.(nus)] .= Inf

#             k = adaptive_k(ps, sort(nus, dims=1))
#         end

#         idxs = [rand(1:N, nbs) for _ in 1:nbs]
#         return new([E(selectdim(ps, ndims(ps), i), q; k=k) for i in idxs], idxs)
#     end
# end
# (bc::BootstrappedExactBhattacharyyaCoefficient)(v::Val{:log}, x) = mean([logbias(e, x[i]) + e(v, x[i]) for (e, i) in zip(bc.ests, bc.idxs)])

# abstract type X; end


#? Probably still need a sort of combined class, but will break out the adaptive kernel.
# struct AdaptiveKernelDistanceEstimator{D, E <: StatisticalDistanceEstimator}
#     # The default kernel options
#     kr::Kernel{D}
#     # The desired ess used to adapt the bandwidth
#     ess::Int
#     # The estimator for our sample
#     est::E
# end
# AdaptiveKernelDistanceEstimator(k::Kernel, ess, e::StatisticalDistanceEstimator) = AdaptiveKernelDistanceEstimator(KernelRecipe(k), ess, e)
# AdaptiveKernelDistanceEstimator(p::ImplicitPosterior, k::KernelRecipe, e::Type{E<:StatisticalDistanceEstimator}, ess::Integer, ths, xs) = AdaptiveKernelDistanceEstimator(k, ess, e(ths, prior(post)))
