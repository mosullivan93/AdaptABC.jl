# The "recipe" methods are those that perform an adaptive sampling step.
function _rejection(mdl::ImplicitPosterior, K::KernelRecipe{Uniform}, N, keep=N)
    θ = Array{eltype(prior(mdl))}(undef, length(prior(mdl)), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)

    Threads.@threads for i = 1:N
        (theta, xs) = rand(mdl.bayesmodel)
        θ[:, i] .= theta
        X[:, i] .= xs
    end
    ρ = colwise(K.ρ, mdl.observed_summs, X)

    idx = getindex(sortperm(ρ), 1:keep)
    return (θ[:, idx], X[:, idx], ρ[idx], Kernel(mdl, ρ[last(idx)], K))
end

function _rejection(mdl::ImplicitPosterior, K::KernelRecipe, N, keep=N)
    θ = Array{eltype(prior(mdl))}(undef, length(prior(mdl)), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)

    Threads.@threads for i = 1:N
        (theta, xs) = rand(mdl.bayesmodel)
        θ[:, i] .= theta
        X[:, i] .= xs
    end
    ρ = colwise(K.ρ, mdl.observed_summs, X)

    u = rand(N)
    h = _inv.(K.d, ρ, u)
    
    idx = getindex(sortperm(h), 1:keep)
    return (θ[:, idx], X[:, idx], ρ[idx], Kernel(mdl, h[last(idx)], K))
end

function _rejection(mdl::ImplicitPosterior, K::PosteriorApproximator{Uniform}, N)
    θ = Array{eltype(prior(mdl))}(undef, length(prior(mdl)), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    Threads.@threads for i = 1:N
        local theta, xs, dist
        local done_sim = false
        while !done_sim
            (theta, xs) = rand(mdl.bayesmodel)
            dist = distance(K, xs)
            done_sim = isfinite(K(xs, Val(:logpdf_prop)))
        end
        θ[:, i] .= theta
        X[:, i] .= xs
        ρ[i] = dist
    end

    idx = sortperm(ρ)
    return (θ[:, idx], X[:, idx], ρ[idx], K)
end

function _rejection(mdl::ImplicitPosterior, K::PosteriorApproximator, N)
    θ = Array{eltype(prior(mdl))}(undef, length(prior(mdl)), N)
    X = Array{eltype(mdl)}(undef, length(mdl), N)
    ρ = Vector{Float64}(undef, N)

    local kmax = K(mdl.observed_summs, Val(:pdf_prop))
    Threads.@threads for i = 1:N
        local theta, xs, dist
        local done_sim = false
        while !done_sim
            (theta, xs) = rand(mdl.bayesmodel)
            dist = distance(K, xs)
            done_sim = rand() ≤ K(xs, Val(:pdf_prop))/kmax
        end
        θ[:, i] .= theta
        X[:, i] .= xs
        ρ[i] = dist
    end

    idx = sortperm(ρ)
    return (θ[:, idx], X[:, idx], ρ[idx], K)
end

# todo: Allow specifying kernel in a convenience wrapper
rejection_sampler(mdl::ImplicitPosterior, K::KernelRecipe, N::Integer) = _rejection(mdl, K, N)
rejection_sampler(mdl::ImplicitPosterior, K::PosteriorApproximator, N::Integer) = _rejection(mdl, K, N)
function rejection_sampler(mdl::ImplicitPosterior, N::Integer;
                           ϵ::Union{Real, Nothing}=nothing,
                           keep_only::Union{Integer, Percentage, Nothing}=isnothing(ϵ) ? %(5) : nothing)

    # Whether we need to check each simulation or will accept all initial draws.
    precompute = isinf((ϵ = something(ϵ, Inf))...)
    # Whether we will return only a subset, or all of the samples.
    keepall = isnothing(keep_only)

    (precompute | keepall) ||
        error("You must specify (only) one of ϵ or keep_only.")

    if precompute
        if keepall
            # Given nothing
            keep = N
        elseif isa(keep_only, Percentage)
            # Given percentage
            keep = keep_only(N)
        else
            # Given desired number
            keep = keep_only
        end

        keep > 0 || error("It does not make sense to only keep a non-positive number of samples.")
        keep <= N || error("It does not make sense to keep more than the number of samples generated.")
        return _rejection(mdl, KernelRecipe(), N, keep)
    else
        return _rejection(mdl, Kernel(mdl, ϵ), N)
    end
end
