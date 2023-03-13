ess(ws::Vector{Float64}) = sum(ws)^2/sum(ws.^2)

function find_incremental_weights(post::ImplicitPosterior, kr::KernelRecipe, xs::Matrix{Float64}, target_ess::Int)
    dists = distance.(kr, post, eachcol(xs))
    h = _inv.(kr.family, dists, rand(length(dists)))

    ub = h[target_ess]
    while ess(_pdfu.(kr, dists/ub)) < target_ess
        ub *= 1.1
    end

    res = Optim.optimize(x -> abs(ess(_pdfu.(kr, dists/x)) - target_ess), 0.0, ub)
    res = Optim.minimizer(res)
    return res
end

function find_incremental_weights(post::ImplicitPosterior, kr::KernelRecipe{Uniform}, xs::Matrix{Float64}, target_ess::Int)
    rhos = distance.(kr, post, eachcol(xs))
    return partialsort(rhos, target_ess)
end

adapt_ess(post::ImplicitPosterior, kr::KernelRecipe, xs::Matrix{Float64}, target_ess::Int) = Kernel(post, find_incremental_weights(post, kr, xs, target_ess), kr)
