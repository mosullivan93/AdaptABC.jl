# todo: always use th_0 and x_0 and make the dispatch/frontend generate them.

function mcmc_sampler(mdl::ImplicitPosterior{M}, K::PosteriorApproximator{Uniform}, N::Integer, q::Distribution; θ₀=nothing, X₀=missing) where {M}
    π = prior(mdl)

    θ = Array{eltype(π)}(undef, length(π), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    # Initialise chain
    local theta, summ, dist
    if isnothing(θ₀)
        theta, summ, (dist,) = vec.(first(rejection_sampler(mdl, 1, K), 3))
    else
        theta = vec(θ₀)
        summ = ismissing(X₀) ? rand(M(theta...)) : vec(X₀)
        dist = distance(K, summ)
    end
 
    local acc = 0
    for i in 1:N
        # todo: The distribution being univariate is a problem with logpdf... see how distributions folks do it.
        # todo: maybe just logpdf.(_, _)
        #? https://github.com/JuliaStats/Distributions.jl/blob/cd45ecc6ab9ed186ad741de41a64b24c5336e4cc/src/univariates.jl#L314
        prop_theta = theta + rand(q)

        if randexp() ≥ (logpdf(π, theta) - logpdf(π, prop_theta))
            prop_summ = rand(M(prop_theta...))
            prop_dist = distance(K, prop_summ)
            if isfinite(K(prop_summ, Val(:logpdf_prop)))
                theta, summ, dist = prop_theta, prop_summ, prop_dist
                acc += 1
            end
        end

        θ[:, i], X[:, i], ρ[i] = theta, summ, dist
    end

    return (θ, X, ρ, acc)
end

function mcmc_sampler(mdl::ImplicitPosterior{M}, K::PosteriorApproximator, N::Integer, q::Distribution; θ₀=nothing, X₀=missing) where {M}
    π = prior(mdl)

    θ = Array{eltype(π)}(undef, length(π), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    # Initialise chain
    local theta, summ, dist
    if isnothing(θ₀)
        theta, summ, (dist,) = vec.(first(rejection_sampler(mdl, 1, K), 3))
    else
        theta = vec(θ₀)
        summ = ismissing(X₀) ? rand(M(theta...)) : vec(X₀)
        dist = distance(K, summ)
    end
 
    local acc = 0
    for i in 1:N
        prop_theta = theta + rand(q)
        prop_summ = rand(M(prop_theta...))
        prop_dist = distance(K, prop_summ)

        if randexp() ≥ ((K(summ, Val(:logpdf_prop)) + logpdf(π, theta)) - (K(prop_summ, Val(:logpdf_prop)) + logpdf(π, prop_theta)))
            theta, summ, dist = prop_theta, prop_summ, prop_dist
            acc += 1
        end

        θ[:, i], X[:, i], ρ[i] = theta, summ, dist
    end

    return (θ, X, ρ, acc)
end
