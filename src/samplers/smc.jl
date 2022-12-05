function smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe{Uniform}, N::Integer;
                     drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
    # Setup
    π = prior(mdl)

    # Setup consts
    N_move = N_drop = drop(N)
    N_keep = N - N_drop
    R = R₀
    
    # Initialise particles
    #* always sorted by this method
    θ, X, ρ, curr_K = rejection_sampler(mdl, K, N)
    @show curr_K.ϵ

    local q_cov
    final_iter = false
    while true
        if !final_iter
            curr_K = adapt(curr_K, ϵ=ρ[N_keep])
            @show curr_K.ϵ
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
        Threads.@threads for I = 1:N_move
            shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves = mcmc_sampler(mdl, curr_K, R, MvNormal(q_cov); θ₀=θ[:, I])

            θ[:, I] .= shuffle_θ[:, end]
            X[:, I] .= shuffle_X[:, end]
            ρ[I] = last(shuffle_ρ)
            accs[I] = shuffle_moves
        end
        final_iter && break
        @show p_acc = sum(accs)/(R*N_drop)
        @show R = max(1, ceil(Int, log(1-p_acc, c)))

        idx = sortperm(ρ)
        θ = θ[:, idx]
        X = X[:, idx]
        ρ = ρ[idx]

        @show mean(θ, dims=2)
        @show var(θ, dims=2)

        final_iter = p_acc < p_thresh
    end

    return (θ, X, ρ, curr_K)
end

## Non-uniform kernels
# smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe, N::Integer; drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
