#! FIX MEDIAN. CONVERT TO OLCM.

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
    local curr_K::Kernel{Uniform, D, ScalingTransform{1, S}, typeof(mdl)}
    local all_K::ProductKernel{Uniform, D, ScalingTransform{1, S}}

    # Compute an excess of particles to determine initial scaling.
    θ, X, ρ, curr_K = rejection_sampler(mdl, K, N)
    @show curr_K.bandwidth
    all_K = ProductKernel(curr_K)
    
    # it's not keen on median...
    next_K = revise(curr_K.recipe, t=ScalingTransform(vec(1 ./ median(abs.(X .- median(X, dims=2)), dims=2)), Val(S)))

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
        local n_sims_stored = min(max_store_iters, R)
        local all_sims = zeros(Float64, length(mdl), n_sims_stored)
        local store_sims = Threads.Atomic{Bool}(true)
        local saved_sims = Threads.Atomic{Int}(1)
            
        let R=R, all_K=all_K, q_cov=q_cov, store_sims=store_sims, n_sims_stored=n_sims_stored #, saved_sims=saved_sims
            # for I = 1:N_move
            Threads.@threads for I = 1:N_move
                shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves, shuffle_uniq_sims = mcmc_sampler(mdl, all_K, R, MvNormal(q_cov); θ₀=θ[:, I], X₀=X[:, I], store_uniq_sims=store_sims[])

                θ[:, I] .= shuffle_θ[:, end]
                X[:, I] .= shuffle_X[:, end]
                ρ[I] = last(shuffle_ρ)
                accs[I] = shuffle_moves

                if store_sims[]
                    # How many uniques did we get back (matters for early rejection)
                    nuniq = size(shuffle_uniq_sims, 2)
                    # What's the next index we need to place them?
                    next_ins_idx = Threads.atomic_add!(saved_sims, nuniq)
                    # Did the last update take us to the end?
                    if next_ins_idx < n_sims_stored
                        # Minimum of how many slots are left vs how many are we adding
                        n_summs_ins = min(n_sims_stored - next_ins_idx + 1, nuniq)
                        # Insert the necessary number of summaries to the records
                        all_sims[:, next_ins_idx:(next_ins_idx+n_summs_ins-1)] = shuffle_uniq_sims[:, 1:n_summs_ins]
                    else
                        # Update the flag that we've stored enough.
                        store_sims[] = false
                    end
                end
            end
        end
        final_iter && break
        @show p_acc = sum(accs)/(R*N_move)
        @show R = max(1, ceil(Int, log(1-p_acc, c)))
        
        n_sims_stored = min(n_sims_stored, saved_sims[])
        mads = median(abs.(all_sims[:, 1:n_sims_stored] .- median(all_sims[:, 1:n_sims_stored], dims=2)), dims=2)
        next_K = revise(curr_K.recipe, t=ScalingTransform(vec(1 ./ mads), Val(S)))

        idx = sortperm(ρ)
        θ .= θ[:, idx]
        X .= X[:, idx]
        ρ .= ρ[idx]

        @show mean(θ, dims=2)
        @show var(θ, dims=2)

        final_iter = p_acc < p_thresh
    end

    return (θ, X, ρ, all_K)
end

## Non-uniform kernels
# smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe, N::Integer; drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
