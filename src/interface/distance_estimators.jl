# todo: Put the estimators into a submodule?
abstract type StatisticalDistanceEstimator; end

abstract type EstimatorInputArgsFormat; end
struct MaskedSamples <: EstimatorInputArgsFormat; end
abstract type WeightedSamples <: EstimatorInputArgsFormat; end
struct LogWeights <: WeightedSamples; end
struct LinearWeights <: WeightedSamples; end

abstract type EstimatorOutputScale; end
struct LinearScale <: EstimatorOutputScale; end
struct LogScale <: EstimatorOutputScale; end

abstract type EstimatorOutputFormat; end
struct ExactAnswer; end
struct BiasedAnswer; end

EstimatorInputArgsFormat(::Type{<:StatisticalDistanceEstimator}) = error("Undefined InputArgsType: Should be MaskedSamples, LogWeights, or LinearWeights.")

# a function that uses the EstimatorInputArgsFormat to return the function that is applied to each sample
estimator_kernel_input(::T, k::Kernel, x) where {T<:StatisticalDistanceEstimator} = estimator_kernel_input(EstimatorInputArgsFormat(T), k, x)
estimator_kernel_input(::MaskedSamples, k, x) = isfinite(logpdf(k, x))
estimator_kernel_input(::LogWeights, k, x) = logpdf(k, x)
estimator_kernel_input(::LinearWeights, k, x) = pdf(k, x)

# a function to determine the output scale and format
EstimatorOutputScale(::Type{<:StatisticalDistanceEstimator}) = error("Undefined output scale: Should be LogScale or LinearScale.")
EstimatorOutputFormat(::Type{<:StatisticalDistanceEstimator}) = error("Undefined output format: Should be BiasedAnswer or ExactAnswer.")

# a function to handle the bias/unbiased evaluation
evaluate_biased(e::T, x) where {T<:StatisticalDistanceEstimator} = evaluate_biased(EstimatorOutputFormat(T), e, x)
evaluate_biased(e::ExactAnswer, ::StatisticalDistanceEstimator, x) = evaluate(EstimatorOutputFormat(T), e, x)

evaluate(e::T, x) where {T<:StatisticalDistanceEstimator} = evaluate(EstimatorOutputFormat(T), e, x)
evaluate(e::BiasedAnswer, ::StatisticalDistanceEstimator, _) = error("Missing implementation for unbiased evaluation.")

# a function to handle log/linear scale evaluation
est(e::T, x) where {T<:StatisticalDistanceEstimator} = est(EstimatorOutputScale(T), e, x)
est(::LinearScale, e::StatisticalDistanceEstimator, x) = evaluate(e, x)
est(::LogScale, e::StatisticalDistanceEstimator, x) = exp(logest(e, x))

estb(e::T, x) where {T<:StatisticalDistanceEstimator} = estb(EstimatorOutputScale(T), e, x)
estb(::LinearScale, e::StatisticalDistanceEstimator, x) = evaluate_biased(e, x)
estb(::LogScale, e::StatisticalDistanceEstimator, x) = exp(logestb(e, x))

logest(e::T, x) where {T<:StatisticalDistanceEstimator} = logest(EstimatorOutputScale(T), e, x)
logest(::LinearScale, e::StatisticalDistanceEstimator, x) = log(est(e, x))
logest(::LogScale, e::StatisticalDistanceEstimator, x) = evaluate(e, x)

logestb(e::T, x) where {T<:StatisticalDistanceEstimator} = logestb(EstimatorOutputScale(T), e, x)
logestb(::LinearScale, e::StatisticalDistanceEstimator, x) = log(estb(e, x))
logestb(::LogScale, e::StatisticalDistanceEstimator, x) = evaluate_biased(e, x)

abstract type AlphaIntegralFunctional{α} <: StatisticalDistanceEstimator; end
Broadcast.broadcastable(e::AdaptABC.StatisticalDistanceEstimator) = Ref(e)

const BhattacharyyaCoefficient = AlphaIntegralFunctional{0.5}

silverman_rule_of_thumb(ps::AbstractVector) = 0.9*min(std(ps; corrected=false), iqr(ps)/1.34)*(length(ps))^(-1/5)
function silverman_rule_of_thumb(ps::AbstractMatrix)
    d, N = size(ps)
    return (4/(d+2))^(1/(d+4))*sqrt(mean(var(ps, dims=2)))*N^(-1/(d+4))
end
adaptive_k(ps, nus) = sum(median(nus, dims=2) .< silverman_rule_of_thumb(ps)) + 1

struct WeightedSampleBC <: BhattacharyyaCoefficient
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
EstimatorInputArgsFormat(::Type{WeightedSampleBC}) = LogWeights()
EstimatorOutputScale(::Type{WeightedSampleBC}) = LogScale()
EstimatorOutputFormat(::Type{WeightedSampleBC}) = BiasedAnswer()
evaluate_biased(e::WeightedSampleBC, lws) = logsumexp((lws .+ e.ts)/2) - logsumexp(lws)/2
evaluate(e::WeightedSampleBC, lws) = evaluate_biased(e, lws) + loggamma(e.k) - loggamma(e.k + 1/2) + (e.d/2*log(π) - loggamma(e.d/2 + 1))/2 - log(e.N)/2

struct SubsetSampleBC <: BhattacharyyaCoefficient
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
EstimatorInputArgsFormat(::Type{SubsetSampleBC}) = MaskedSamples()
EstimatorOutputScale(::Type{SubsetSampleBC}) = LogScale()
EstimatorOutputFormat(::Type{SubsetSampleBC}) = BiasedAnswer()
evaluate(e::SubsetSampleBC, idx) = evaluate_biased(e, idx) + loggamma(e.k) - loggamma(e.k + 1/2) + (e.d/2*log(π) - loggamma(e.d/2 + 1))/2 - log(length(first(to_indices(e.lqs, (idx,)))))
function evaluate_biased(e::SubsetSampleBC, idx)
    kept_nus = e.nus[idx, idx]
    ndups = count(isinf, kept_nus, dims=2)
    lλs = log.(size(kept_nus, 2) .- ndups)

    sort!(kept_nus, dims=1, alg=PartialQuickSort(e.k))
    lηs = log.(kept_nus[e.k, :])

    return logsumexp((e.lqs[idx] .+ lλs .+ e.d*lηs)/2)
end
