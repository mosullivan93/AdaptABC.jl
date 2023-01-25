# todo: Should this be implementing DensityInterface instead?

"Represents a model which can only be simulated for a given parameter configuration"
abstract type ImplicitDistribution <: Distribution{Multivariate, Continuous} end
Broadcast.broadcastable(d::ImplicitDistribution) = Ref(d)
# todo: In future, review whether Continuous makes sense here. I note flow on effects to (log)?pdf
summarise(::Type{<:ImplicitDistribution}, x) = identity(x)
summarise(::T, x) where {T <: ImplicitDistribution} = summarise(T, x)
simulate(::Type{<:ImplicitDistribution}, _)  = error("Not implemented")
simulate(mdl::T) where {T<:ImplicitDistribution}  = simulate(T, params(mdl))
sim_summ(M::Type{<:ImplicitDistribution}, θ) = summarise(M, simulate(M, θ))
sim_summ(::T, x) where {T <: ImplicitDistribution} = sim_summ(T, x)
Base.rand(mdl::T) where {T <: ImplicitDistribution} = sim_summ(mdl, params(mdl))

paramnames(::T) where {T <: ImplicitDistribution} = fieldnames(T)
Distributions.params(mdl::T) where {T <: ImplicitDistribution} = Tuple(getproperty(mdl, f) for f in fieldnames(T))
Distributions.pdf(m::M, x::AbstractVector{Float64}) where {M <: ImplicitDistribution} = IntractableLikelihood(x, m)
Distributions.logpdf(m::M, x::AbstractVector{Float64}) where {M <: ImplicitDistribution} = log(pdf(m, x))

"Represents the joint posterior from assignment of a prior distribution to the implicitly defined likelihood"
abstract type AbstractImplicitBayesianModel{M <: ImplicitDistribution, P<:Distribution} <: Distribution{Multivariate, Continuous}; end
(m::AbstractImplicitBayesianModel)(obs) = ImplicitPosterior(m; obs_data=obs)
(m::AbstractImplicitBayesianModel)(;kwargs...) = ImplicitPosterior(m; kwargs...)
Broadcast.broadcastable(d::AbstractImplicitBayesianModel) = Ref(d)
function Base.rand(mdl::AbstractImplicitBayesianModel)
    pr = rand(prior(mdl))
    xs = simulator(mdl)(pr)

    return (pr, xs)
end
struct ImplicitBayesianModel{M, P} <: AbstractImplicitBayesianModel{M, P}
    model::Type{M}
    prior::P
end
paramnames(mdl::ImplicitBayesianModel) = fieldnames(mdl.model)
prior(mdl::ImplicitBayesianModel) = mdl.prior
simulator(::ImplicitBayesianModel{M}) where {M} = Base.Fix1(sim_summ, M)
Distributions.pdf(p::ImplicitBayesianModel, x) = pdf(p.prior, x[1])*pdf(p.model(x[1]...), x[2])
Distributions.logpdf(p::ImplicitBayesianModel, x) = logpdf(p.prior, x[1]) + logpdf(p.model(x[1]...), x[2])

struct TransformedImplicitBayesianModel{M, P, O, B} <: AbstractImplicitBayesianModel{M, P}
    original_model::O
    t_prior::P
    transform::B

    function TransformedImplicitBayesianModel(m::ImplicitBayesianModel{M, P}, b::B) where {M, P, B<:Bijector}
        tp = transformed(prior(m))
        return new{M, typeof(tp), ImplicitBayesianModel{M, P}, B}(m, tp, b)
    end
end
paramnames(mdl::TransformedImplicitBayesianModel) = paramnames(mdl.original_model)
#* should this be precomputed and saved?
# prior(mdl::TransformedImplicitBayesianModel) = transformed(prior(mdl.original_model), mdl.transform)
prior(mdl::TransformedImplicitBayesianModel) = mdl.t_prior
simulator(mdl::TransformedImplicitBayesianModel{M}) where {M} = simulator(mdl.original_model) ∘ Base.Fix1(invlink, mdl)
Distributions.pdf(p::TransformedImplicitBayesianModel, y) = pdf(prior(p), y[1])*pdf(p.model(invlink(p, x[1])...), x[2])
Distributions.logpdf(p::TransformedImplicitBayesianModel, y) = logpdf(prior(p), y[1]) + logpdf(p.model(invlink(p, x[1])...), x[2])
Bijectors.bijector(m::ImplicitBayesianModel) = bijector(prior(m))
Bijectors.transformed(m::ImplicitBayesianModel, b::Bijector) = TransformedImplicitBayesianModel(m, b)
Bijectors.link(m::TransformedImplicitBayesianModel, x) = m.transform(x)
Bijectors.invlink(m::TransformedImplicitBayesianModel, y) = inverse(m.transform)(y)

"An implicit bayesian model where the observed data has been specified."
#! todo: test when this was the abstraction again... seems like it worked.
struct ImplicitPosterior{M, P, S} <: Distribution{Multivariate, Continuous}
    bayesmodel::Union{ImplicitBayesianModel{M, P}, TransformedImplicitBayesianModel{M, P}} #! fix this for type stability. Shouldn't be Abstract.
    # bayesmodel::AbstractImplicitBayesianModel{M, P} #! fix this for type stability. Shouldn't be Abstract.
    # observed_data won't be type stable but shouldn't be accessed often.
    observed_data::Union{Missing, Array{Float64}}
    observed_summs::Vector{Float64}

    function ImplicitPosterior(bm::AbstractImplicitBayesianModel{M, P}; obs_data=missing, obs_summs=missing) where {M, P}
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
