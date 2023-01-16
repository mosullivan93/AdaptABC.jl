function adaptive_smc_sampler_MAD(mdl::ImplicitPosterior{M}, K::KernelRecipe{Uniform}, N::Integer;
                                  drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
    # Setup
    π = prior(mdl)

    # Setup consts
    N_move = N_drop = drop(N)
    N_keep = N - N_drop
    max_store_iters = R = R₀
    
    # Initialise particles
    # Compute an excess of particles to determine initial scaling.
    θ, X, ρ, curr_K = rejection_sampler(mdl, K, N)
    @show curr_K.ϵ
    all_K = ProductKernel(curr_K)
    
    next_K = KernelRecipe(curr_K, T=ScalingTransform(vec(1 ./ median(abs.(X .- median(X, dims=2)), dims=2))))

    local q_cov
    final_iter = false
    while true
        if !final_iter
            # Recompute distances with updated scaling & store new threshold
            ρ .= distance.(next_K, mdl, eachcol(X))

            idx = sortperm(ρ)
            θ = θ[:, idx]
            X = X[:, idx]
            ρ = ρ[idx]

            curr_K = Kernel(mdl, ρ[N_keep], next_K)
            @show curr_K.ϵ
            @show pa_length(curr_K)
            all_K = ProductKernel(all_K, curr_K)

            q_cov = cov(θ[:, 1:N_keep], dims=2)

            #* Put kept samples at end and iterate 1:N_drop
            resample_idx = [rand(1:N_keep, N_drop)..., 1:N_keep...]
            θ .= θ[:, resample_idx]
            X .= X[:, resample_idx]
            ρ .= ρ[resample_idx]
        else
            q_cov = cov(θ, dims=2)
            N_move = N
        end

        local accs = zeros(N_move)
        # todo: Convert to use 1:max_store_iters, adapt R, max_store_iters:R
        local all_sims = zeros(length(mdl), N_move, min(max_store_iters, R))
        q = MvNormal(q_cov)

        Threads.@threads for I = 1:N_move
            theta = θ[:, I]
            summ = X[:, I]
            dist = ρ[I]

            for J = 1:R
                prop_theta = theta + rand(q)

                if randexp() ≥ (logpdf(π, theta) - logpdf(π, prop_theta))
                    store_summ = prop_summ = rand(M(prop_theta...))
                    prop_dist = distance(curr_K, prop_summ)
                    if isfinite(all_K(prop_summ, Val(:logpdf_prop)))
                        theta, summ, dist = prop_theta, prop_summ, prop_dist
                        accs[I] += 1
                    end
                else
                    # Keep the existing statistics if we jump outside of the prior
                    store_summ = summ
                end

                if J <= max_store_iters
                    all_sims[:, I, J] = store_summ
                end
            end

            θ[:, I] .= theta
            X[:, I] .= summ
            ρ[I] = dist
        end
        final_iter && break
        @show p_acc = sum(accs)/(R*N_move)
        @show R = max(1, ceil(Int, log(1-p_acc, c)))
        
        all_sims = reshape(all_sims, (length(mdl), :))
        next_K = KernelRecipe(curr_K, T=ScalingTransform(vec(1 ./ median(abs.(all_sims .- median(all_sims, dims=2)), dims=2))))

        idx = sortperm(ρ)
        θ = θ[:, idx]
        X = X[:, idx]
        ρ = ρ[idx]

        @show mean(θ, dims=2)
        @show var(θ, dims=2)

        final_iter = p_acc < p_thresh
    end

    return (θ, X, ρ, all_K)
end

## Non-uniform kernels
# smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe, N::Integer; drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
