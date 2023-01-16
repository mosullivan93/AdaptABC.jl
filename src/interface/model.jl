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
struct ImplicitBayesianModel{M <: ImplicitDistribution, P<:Distribution} <: Distribution{Multivariate, Continuous}
    model::Type{M}
    prior::P
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
(m::ImplicitBayesianModel)(obs) = ImplicitPosterior(m; obs_data=obs)
(m::ImplicitBayesianModel)(;kwargs...) = ImplicitPosterior(m; kwargs...)

"An implicit bayesian model where the observed data has been specified."
struct ImplicitPosterior{M, P, S} <: Distribution{Multivariate, Continuous}
    bayesmodel::ImplicitBayesianModel{M, P}
    # observed_data won't be type stable but shouldn't be accessed often.
    observed_data::Union{Missing, Array{Float64}}
    observed_summs::Vector{Float64}

    function ImplicitPosterior(bm::ImplicitBayesianModel{M, P}; obs_data=missing, obs_summs=missing) where {M, P}
        if ismissing(obs_summs)
            ismissing(obs_data) && error("No observations given. Do you mean to use ImplicitBayesianModel?")
            obs_summs = summarise(M, obs_data)
        else
            ismissing(obs_data) || @warn("This constructor does not validate that summarise(obs_data) is equal to obs_summs.")
        end
        return new{M, P, length(obs_summs)}(bm, obs_data, obs_summs)
    end
end
ImplicitPosterior(m::Type{M}, p::Distribution; kwargs...) where {M <: ImplicitDistribution} = ImplicitPosterior(ImplicitBayesianModel(m, p); kwargs...)
Base.length(_::ImplicitPosterior{M, P, S} where {M, P}) where {S} = S
paramnames(post::ImplicitPosterior) = paramnames(post.bayesmodel)
prior(post::ImplicitPosterior) = prior(post.bayesmodel)
Distributions.pdf(p::ImplicitPosterior, th::Vector{Float64}) = pdf(p.bayesmodel, (th, p.observed_data))
Distributions.logpdf(p::ImplicitPosterior, th::Vector{Float64}) = logpdf(p.bayesmodel, (th, p.observed_data))
