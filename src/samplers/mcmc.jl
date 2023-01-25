# todo: always use th_0 and x_0 and make the dispatch/frontend generate them.

function vecunpack(t, ::Val{n}) where {n}
    return ntuple(i -> vec(t[i]), Val(n))
end

function mcmc_sampler(mdl::ImplicitPosterior{M}, K::PosteriorApproximator{Uniform}, N::Integer, q::Distribution, init::NTuple{3, Vector{Float64}}) where {M}
    π = prior(mdl)
    sim_fn = simulator(mdl.bayesmodel)

    θ = Array{eltype(π)}(undef, length(π), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    # Initialise chain
    local theta::Vector{Float64}, summ::Vector{Float64}, (dist::Float64,) = init
 
    local acc = 0
    for i in 1:N
        # todo: The distribution being univariate is a problem with logpdf... see how distributions folks do it.
        # todo: maybe just logpdf.(_, _)
        #? https://github.com/JuliaStats/Distributions.jl/blob/cd45ecc6ab9ed186ad741de41a64b24c5336e4cc/src/univariates.jl#L314
        prop_theta = theta + rand(q)

        if randexp() ≥ (logpdf(π, theta) - logpdf(π, prop_theta))
            prop_summ = sim_fn(prop_theta)
            prop_dist = distance(K, prop_summ)
            if isfinite(logpdfu(K, prop_summ))
                theta, summ, dist = prop_theta, prop_summ, prop_dist
                acc += 1
            end
        end

        θ[:, i], X[:, i], ρ[i] = theta, summ, dist
    end

    return (θ, X, ρ, acc)
end

function mcmc_sampler(mdl::ImplicitPosterior{M}, K::PosteriorApproximator, N::Integer, q::Distribution, init::NTuple{3, Vector{Float64}}) where {M}
    π = prior(mdl)
    sim_fn = simulator(mdl.bayesmodel)

    θ = Array{eltype(π)}(undef, length(π), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    # Initialise chain
    local theta::Vector{Float64}, summ::Vector{Float64}, (dist::Float64,) = init
 
    local acc::Int64 = 0
    for i in 1:N
        prop_theta = theta + rand(q)
        prop_summ = sim_fn(prop_theta)
        prop_dist = distance(K, prop_summ)

        if randexp() ≥ ((logpdfu(K, summ) + logpdf(π, theta)) - (logpdfu(K, prop_summ) + logpdf(π, prop_theta)))
            theta, summ, dist = prop_theta, prop_summ, prop_dist
            acc += 1
        end

        θ[:, i], X[:, i], ρ[i] = theta, summ, dist
    end

    return (θ, X, ρ, acc)
end

function mcmc_sampler(post::ImplicitPosterior{M}, K::PosteriorApproximator, N::Integer, q::Distribution; θ₀=nothing, X₀=missing) where {M}
    local theta::Vector{Float64}, summ::Vector{Float64}, dist::Vector{Float64}
    if isnothing(θ₀)
        theta, summ, dist = vecunpack(rejection_sampler(post, K, 1), Val(3))
    else
        theta = vec(θ₀)
        summ = ismissing(X₀) ? simulator(mdl.bayesmodel)(theta) : vec(X₀)
        dist = Float64[distance(K, summ)]
    end

    return mcmc_sampler(post, K, N, q, (theta, summ, dist))
end
