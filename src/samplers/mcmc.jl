function vecunpack(t, ::Val{n}) where {n}
    return ntuple(i -> vec(t[i]), Val(n))
end

function mcmc_sampler(mdl::ImplicitPosterior{M}, K::PosteriorApproximator{Uniform}, N::Integer, q::Distribution, init::NTuple{3, Vector{Float64}}, store_sims::Union{Val{true}, Val{false}}=Val(false)) where {M}
    π = prior(mdl)
    sim_fn = simulator(mdl.bayesmodel)

    θ = Array{eltype(π)}(undef, length(π), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    # Initialise chain
    local theta::Vector{Float64}, summ::Vector{Float64}, (dist::Float64,) = init

    local store_uniq_sims::Bool = store_sims isa Val{true}
    X_all = Array{eltype(mdl)}(undef, length(mdl), ifelse(store_uniq_sims, N, 0))
 
    local acc = 0
    local sim_counter = 0
    for i in 1:N
        # todo: The distribution being univariate is a problem with logpdf... see how distributions folks do it.
        # todo: maybe just logpdf.(_, _)
        #? https://github.com/JuliaStats/Distributions.jl/blob/cd45ecc6ab9ed186ad741de41a64b24c5336e4cc/src/univariates.jl#L314
        # maybe use rand(q, 1)?
        prop_theta = theta + rand(q)

        if randexp() ≥ (logpdf.(π, theta) - logpdf.(π, prop_theta))
            prop_summ = sim_fn(prop_theta)
            prop_dist = distance(K, prop_summ)
            if store_uniq_sims
                sim_counter += 1
                X_all[:, sim_counter] .= prop_summ
            end

            if isfinite(logpdfu(K, prop_summ))
                theta, summ, dist = prop_theta, prop_summ, prop_dist
                acc += 1
            end
        end

        θ[:, i], X[:, i], ρ[i] = theta, summ, dist
    end

    return (θ, X, ρ, acc, X_all[:, 1:sim_counter])
end

function mcmc_sampler(mdl::ImplicitPosterior{M}, K::PosteriorApproximator, N::Integer, q::Distribution, init::NTuple{3, Vector{Float64}}, store_sims::Union{Val{true}, Val{false}}=Val(false)) where {M}
    π = prior(mdl)
    sim_fn = simulator(mdl.bayesmodel)

    X_all = Array{eltype(mdl)}(undef, length(mdl), ifelse(store_uniq_sims, N, 0))
    θ = Array{eltype(π)}(undef, length(π), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    # Initialise chain
    local theta::Vector{Float64}, summ::Vector{Float64}, (dist::Float64,) = init

    local store_uniq_sims::Bool = store_sims isa Val{true}
    X_all = Array{eltype(mdl)}(undef, length(mdl), ifelse(store_uniq_sims, N, 0))
 
    local acc::Int64 = 0
    for i in 1:N
        prop_theta = theta + rand(q)
        prop_summ = sim_fn(prop_theta)
        prop_dist = distance(K, prop_summ)
        if store_uniq_sims
            X_all[:. i] .= prop_summ
        end

        if randexp() ≥ ((logpdfu(K, summ) + logpdf(π, theta)) - (logpdfu(K, prop_summ) + logpdf(π, prop_theta)))
            theta, summ, dist = prop_theta, prop_summ, prop_dist
            acc += 1
        end

        θ[:, i], X[:, i], ρ[i] = theta, summ, dist
    end

    return (θ, X, ρ, acc, X_all)
end

function mcmc_sampler(post::ImplicitPosterior{M}, K::PosteriorApproximator, N::Integer, q::Distribution; θ₀=nothing, X₀=missing, store_uniq_sims::Bool=false) where {M}
    local theta::Vector{Float64}, summ::Vector{Float64}, dist::Vector{Float64}
    if isnothing(θ₀)
        theta, summ, dist = vecunpack(rejection_sampler(post, K, 1), Val(3))
    else
        theta = vec(θ₀)
        summ = ismissing(X₀) ? simulator(post.bayesmodel)(theta) : vec(X₀)
        dist = Float64[distance(K, summ)]
    end

    return mcmc_sampler(post, K, N, q, (theta, summ, dist), ifelse(store_uniq_sims, Val{true}(), Val{false}()))
end
