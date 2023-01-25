function adaptive_smc_sampler_MAD(mdl::ImplicitPosterior{M, P, S}, K::KernelRecipe{Uniform, D, T}, N::Integer;
                                  drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M, P, S, D, T}
    # Setup
    π = prior(mdl)
    sim_fn = simulator(mdl.bayesmodel)

    # Setup consts
    local N_drop::Int = drop(N)
    local N_keep::Int = N - N_drop
    local N_move::Int = N_drop
    local R::Int = R₀
    local max_store_iters::Int = R₀
    
    # Initialise particles
    local θ::Matrix{Float64}
    local X::Matrix{Float64}
    local ρ::Vector{Float64}
    local next_K::KernelRecipe{Uniform, D, ScalingTransform{1, S}}
    local curr_K::Kernel{Uniform, D, ScalingTransform{1, S}, ImplicitPosterior{M, P, S}}
    local all_K::ProductKernel{Uniform, D, ScalingTransform{1, S}, ImplicitPosterior{M, P, S}}

    # Compute an excess of particles to determine initial scaling.
    θ, X, ρ, curr_K = rejection_sampler(mdl, K, N)
    @show curr_K.bandwidth
    all_K = ProductKernel(curr_K)
    
    next_K = revise(curr_K.recipe, t=ScalingTransform(vec(1 ./ median(abs.(X .- median(X, dims=2)), dims=2))))

    local q_cov::Matrix{Float64} = Matrix{Float64}(undef, length(π), length(π))
    local p_acc::Float64 = 1.0
    final_iter = false
    while true
        if final_iter
            N_move = N
        else
            # Recompute distances with updated scaling
            ρ .= distance.(next_K, mdl, eachcol(X))
            idx = sortperm(ρ)

            #* Put kept samples at end and iterate 1:N_drop
            resample_idx = idx[vcat(rand(1:N_keep, N_drop), 1:N_keep)]
            θ .= θ[:, resample_idx]
            X .= X[:, resample_idx]
            ρ .= ρ[resample_idx]

            curr_K = Kernel(mdl, last(ρ), next_K)
            @show curr_K.bandwidth
            @show pa_length(curr_K)
            all_K = ProductKernel(all_K, curr_K)
        end

        q_cov .= cov(θ[:, 1:N_move], dims=2)
        local accs = zeros(Int, N_move)
        # todo: Convert to use 1:max_store_iters, adapt R, max_store_iters:R
        local all_sims = zeros(Float64, length(mdl), N_move, min(max_store_iters, R))
        q = MvNormal(q_cov)

        Threads.@threads for I = 1:N_move
            theta = θ[:, I]
            summ = X[:, I]
            dist = ρ[I]

            for J = 1:R
                prop_theta = theta + rand(q)

                if randexp() ≥ (logpdf(π, theta) - logpdf(π, prop_theta))
                    store_summ = prop_summ = sim_fn(prop_theta)
                    prop_dist = distance(curr_K, prop_summ)
                    if isfinite(logpdfu(all_K, prop_summ))
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
        next_K = revise(curr_K.recipe, t=ScalingTransform(vec(1 ./ median(abs.(all_sims .- median(all_sims, dims=2)), dims=2))))

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
