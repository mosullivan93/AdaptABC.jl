function choose_weights_oaat(mdl::ImplicitPosterior, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Integer=5)
    d, N = size(th)
    Nx = length(mdl)

    nus = zeros((N, N))
    pairwise!(nus, euclidean, th)
    lQs = logpdf(prior(mdl), th)

    function compute_bc(ind)
        ind_K = adapt(K, T=ScalingTransform(1:Nx .== ind))
        rhos = distance.(ind_K, mdl, eachcol(xs))
        eps_ind = partialsort(rhos, N_keep)
        keep_idx = findall(rhos .<= eps_ind)[1:N_keep]
        kept_nus = nus[keep_idx, keep_idx]

        ndups = sum(iszero, kept_nus, dims=2)
        sort!(kept_nus, dims=2, alg=PartialQuickSort(k:k+maximum(ndups)))
        nks = view(kept_nus, CartesianIndex.(enumerate(k .+ ndups)))
        lNs = d*log.(nks)

        return logsumexp((log.(N_keep .- ndups) .+ lQs[keep_idx] .+ lNs)/2), ind_K, eps_ind, keep_idx
    end

    # return minimum(compute_bc(i) for i = 1:Nx)
    Ls = [compute_bc(i) for i = 1:Nx]
    return minimum(Ls)
end

function adaptive_smc_sampler_opt_oaat(mdl::ImplicitPosterior{M}, K::KernelRecipe{Uniform}, N::Integer;
                                       drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05, k=5) where {M}
    # Setup
    π = prior(mdl)

    # Setup consts
    N_move = N_drop = drop(N)
    N_keep = N - N_drop
    R = R₀
    
    # Initialise particles
    # Compute an excess of particles to determine initial scaling.
    θ, X, ρ, curr_K = rejection_sampler(mdl, K, N)
    @show curr_K.ϵ
    all_K = ProductKernel(curr_K)

    local q_cov
    final_iter = false
    while true
        if !final_iter            
            # adapt the next kernel
            _, next_K, eps_new, keep_idx = choose_weights_oaat(mdl, K, θ, X, N_keep, k)
            curr_K = Kernel(mdl, eps_new, next_K)
            @show curr_K.ϵ
            @show pa_length(curr_K)
            all_K = ProductKernel(all_K, curr_K)

            # Put the kept particles at the end.
            idx = vcat(sample(keep_idx, N_drop), keep_idx)
            θ = θ[:, idx]
            X = X[:, idx]
            ρ = ρ[idx]

            q_cov = cov(θ[:, N_move+1:end], dims=2)
        else
            q_cov = cov(θ, dims=2)
            N_move = N
        end

        local accs = zeros(N_move)
        Threads.@threads for I = 1:N_move
            shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves = mcmc_sampler(mdl, all_K, R, MvNormal(q_cov); θ₀=θ[:, I])

            θ[:, I] .= shuffle_θ[:, end]
            X[:, I] .= shuffle_X[:, end]
            ρ[I] = last(shuffle_ρ)
            accs[I] = shuffle_moves
        end
        final_iter && break
        @show p_acc = sum(accs)/(R*N_move)
        @show R = max(1, ceil(Int, log(1-p_acc, c)))

        # idx = sortperm(ρ)
        # θ = θ[:, idx]
        # X = X[:, idx]
        # ρ = ρ[idx]

        @show mean(θ, dims=2)
        @show var(θ, dims=2)

        final_iter = p_acc < p_thresh
    end

    return (θ, X, ρ, all_K)
end

## Non-uniform kernels
# smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe, N::Integer; drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
