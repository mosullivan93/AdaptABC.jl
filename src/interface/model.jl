# todo: Should this be implementing DensityInterface instead?

"Represents a model which can only be simulated for a given parameter configuration"
abstract type ImplicitDistribution <: Distribution{Multivariate, Continuous} end
Broadcast.broadcastable(d::ImplicitDistribution) = Ref(d)
# todo: In future, review whether Continuous makes sense here. I note flow on effects to (log)?pdf
summarise(::Type{<:ImplicitDistribution}, x) = identity(x)
summarise(::T, x) where {T <: ImplicitDistribution} = summarise(T, x)
simulate(::Type{<:ImplicitDistribution}, _)  = error("Not implemented")
simulate(mdl::T) where {T<:ImplicitDistribution}  = simulate(T, params(mdl))
Base.rand(mdl::T) where {T <: ImplicitDistribution} = summarise(T, simulate(T, params(mdl)))

paramnames(::T) where {T <: ImplicitDistribution} = fieldnames(T)
Distributions.params(mdl::T) where {T <: ImplicitDistribution} = Tuple(getproperty(mdl, f) for f in fieldnames(T))
Distributions.pdf(m::M, x::AbstractVector{Float64}) where {M <: ImplicitDistribution} = IntractableLikelihood(x, m)
Distributions.logpdf(m::M, x::AbstractVector{Float64}) where {M <: ImplicitDistribution} = log(pdf(m, x))

"Represents the joint posterior from assignment of a prior distribution to the implicitly defined likelihood"
struct ImplicitBayesianModel{M} <: Distribution{Multivariate, Continuous}
    model::Type{M}
    prior

    ImplicitBayesianModel(m::Type{M}, p::Distribution) where {M <: ImplicitDistribution} = new{M}(m, p)
end
Broadcast.broadcastable(d::ImplicitBayesianModel) = Ref(d)
paramnames(mdl::ImplicitBayesianModel) = fieldnames(mdl.model)
prior(mdl::ImplicitBayesianModel) = mdl.prior
function Base.rand(mdl::ImplicitBayesianModel)
    pr = rand(mdl.prior)
    xs = rand(mdl.model(pr...))

    return (pr, xs)
end
Distributions.pdf(p::ImplicitBayesianModel, x) = pdf(p.prior, x[1])*pdf(p.model(x[1]...), x[2])
Distributions.logpdf(p::ImplicitBayesianModel, x) = logpdf(p.prior, x[1]) + logpdf(p.model(x[1]...), x[2])
(m::ImplicitBayesianModel{M})(obs) where {M} = ImplicitPosterior{M}(m; obs_data=obs)
(m::ImplicitBayesianModel{M})(;kwargs...) where {M} = ImplicitPosterior{M}(m; kwargs...)

"An implicit bayesian model where the observed data has been specified."
struct ImplicitPosterior{M} <: Distribution{Multivariate, Continuous}
    bayesmodel::ImplicitBayesianModel{M}
    observed_data
    observed_summs

    function ImplicitPosterior{M}(bm::ImplicitBayesianModel{M}; obs_data=missing, obs_summs=missing) where {M}
        if ismissing(obs_summs)
            ismissing(obs_data) && error("Not observations given...")
            obs_summs = summarise(M, obs_data)
        else
            ismissing(obs_data) || @warn("This constructor does not validate that summarise(obs_data) is equal to obs_summs.")
        end
        return new{M}(bm, obs_data, obs_summs)
    end
end
Broadcast.broadcastable(d::ImplicitPosterior) = Ref(d)
ImplicitPosterior(m::Type{M}, p::Distribution; kwargs...) where {M <: ImplicitDistribution} = ImplicitPosterior{M}(ImplicitBayesianModel(m, p); kwargs...)
Base.length(post::ImplicitPosterior) = length(post.observed_summs)
paramnames(post::ImplicitPosterior) = paramnames(post.bayesmodel)
prior(post::ImplicitPosterior) = prior(post.bayesmodel)
Distributions.pdf(p::ImplicitPosterior, th::AbstractVector{Float64}) = pdf(p.bayesmodel, (th, p.observed_data))
Distributions.logpdf(p::ImplicitPosterior, th::AbstractVector{Float64}) = logpdf(p.bayesmodel, (th, p.observed_data))
