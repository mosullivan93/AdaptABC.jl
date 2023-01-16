function choose_weights_nm(mdl::ImplicitPosterior, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Int, k::Union{Missing, Int}=missing)
    Nx = length(mdl)
    M = Sphere(Nx - 1)

    # bc = SubsetSampleBC(th, prior(mdl); k=k)
    bc = WeightedSampleBC(th, prior(mdl); k=k)

    function compute_bc(ws)
        ind_K = adapt(K, T=ScalingTransform(ws))
        rhos = distance.(ind_K, mdl, eachcol(xs))
        eps_ind = partialsort(rhos, N_keep)
        ind_ws = rhos .<= eps_ind
        keep_idx = findall(ind_ws)[1:N_keep]

        L = bc(Val(:log), ind_ws)
        # L = bc(Val(:log), isfinite.(ind_ws))
        return L, ind_K, eps_ind, keep_idx
    end
    man_obj(::AbstractManifold, p) = first(compute_bc(p))

    mopt_res = NelderMead(M, man_obj, eachrow(diagm(ones(Nx))))
    # mopt_res = particle_swarm(M, man_obj)
    return compute_bc(abs.(mopt_res))

    # ntests = 1_000
    # rand_ws = rand(M, ntests)
    # cmp_ls = zeros(ntests)
    # Threads.@threads for I = 1:ntests
    #     cmp_ls[I] = man_obj(M, rand_ws[I])
    # end

    # min_idx = argmin(cmp_ls)
    # return compute_bc(abs.(rand_ws[min_idx]))
end

function adaptive_smc_sampler_opt_dfo(mdl::ImplicitPosterior{M}, K::KernelRecipe{Uniform}, N::Integer;
                                      drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05, k::Union{Missing, Int}=missing) where {M}
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
            _, next_K, eps_new, keep_idx = choose_weights_nm(mdl, K, θ, X, N_keep, k)
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
