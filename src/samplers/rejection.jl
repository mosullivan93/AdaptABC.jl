# The "recipe" methods are those that perform an adaptive sampling step.
function _rejection(mdl::ImplicitPosterior, K::KernelRecipe{Uniform}, N, keep=N)
    parts = Vector{Particle}(undef, N)

    Threads.@threads for i = 1:N
        (theta, xs) = rand(mdl.bayesmodel)
        parts[i] = Particle(theta, xs, distance(K, mdl, xs))
    end
    idx = partialsortperm(dist.(parts), 1:keep);

    return (parts[idx], Kernel(mdl, dist(parts[last(idx)]), K))
end

function _rejection(mdl::ImplicitPosterior, K::KernelRecipe, N, keep=N)
    parts = Vector{Particle}(undef, N)

    Threads.@threads for i = 1:N
        (theta, xs) = rand(mdl.bayesmodel)
        parts[i] = Particle(theta, xs, distance(K, mdl, xs))
    end

    h = _inv.(K.family, dist.(parts), rand(N))
    idx = partialsortperm(h, 1:keep)

    return (parts[idx], Kernel(mdl, h[last(idx)], K))
end

function _rejection(mdl::ImplicitPosterior, K::PosteriorApproximator{Uniform}, N)
    parts = Vector{Particle}(undef, N)

    Threads.@threads for i = 1:N
        local theta, xs
        while true
            (theta, xs) = rand(mdl.bayesmodel)
            isfinite(logpdfu(K, xs)) && break
        end
        parts[i] = Particle(theta, xs, distance(K, xs))
    end

    sort!(parts; by=dist)
    return parts
end

function _rejection(mdl::ImplicitPosterior, K::PosteriorApproximator, N)
    parts = Vector{Particle}(undef, N)

    local kmax = _pdfu(K, 0.0)
    Threads.@threads for i = 1:N
        local theta, xs
        while true
            (theta, xs) = rand(mdl.bayesmodel)
            (rand() ≤ pdfu(K, xs)/kmax) && break
        end
        parts[i] = Particle(theta, xs, distance(K, xs))
    end

    sort!(parts; by=dist)
    return parts
end

# todo: Allow specifying kernel in a convenience wrapper
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
        return _rejection(mdl, Kernel(mdl, ϵ, KernelRecipe()), N)
    end
end
