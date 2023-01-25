function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
    M = Manifolds.Sphere(S - 1)
    adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
    # adaptive_estimator = AdaptiveKernelEstimator(WeightedSampleBC, post, th, xs, K, N_keep; k=k)
    scales = vec(std(xs, dims=2))

    man_obj(::AbstractManifold, p) = first(logestb(adaptive_estimator, ScalingTransform(p./scales)))
    mopt_res = Manopt.NelderMead(M, man_obj, NelderMeadSimplex(collect(eachrow(diagm(ones(S))))))
    # mopt_res = Manopt.NelderMead(M, man_obj)
    # mopt_res = particle_swarm(M, man_obj)
    return logestb(adaptive_estimator, ScalingTransform(abs.(mopt_res)./scales, Val(S)))
end

adaptive_smc_sampler_opt_dfo(post, K::KernelRecipe, N::Integer; drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05, k::Union{Int, Missing}=missing) = adaptive_smc_sampler_opt_dfo(post, K, N, drop, R₀, p_thresh, c, k)

function adaptive_smc_sampler_opt_dfo(post::ImplicitPosterior{M, P, S}, K::KernelRecipe{Uniform, D, T}, N::Int,
                                       drop::Percentage=%(50), R₀::Int=10, p_thresh::Float64=0.05, c::Float64=0.05, k::Union{Int, Missing}=missing) where {M, P, S, D, T}
    # Setup
    π = prior(post)

    # Setup consts
    local N_drop::Int = drop(N)
    local N_keep::Int = N - N_drop
    local N_move::Int = N_drop
    local R::Int = R₀
    
    # Initialise particles
    local θ::Matrix{Float64}
    local X::Matrix{Float64}
    local ρ::Vector{Float64}
    local curr_K::Kernel{Uniform, D, ScalingTransform{1, S}, ImplicitPosterior{M, P, S}}
    local all_K::ProductKernel{Uniform, D, ScalingTransform{1, S}, ImplicitPosterior{M, P, S}}
    local pas::Vector{Float64} = Vector{Float64}(undef, S)

    # Compute an excess of particles to determine initial scaling.
    θ, X, ρ, curr_K = rejection_sampler(post, K, N)
    pas .= pa_length(curr_K)
    @show pas
    all_K = ProductKernel(curr_K)

    local keep_idx::Vector{Int} = Vector{Int}(undef, N_keep)
    local mask_idx::Vector{Float64} = Vector{Int}(undef, N)
    local q_cov::Matrix{Float64} = Matrix{Float64}(undef, length(π), length(π))
    local p_acc::Float64 = 1.0
    final_iter = false
    while true
        if final_iter
            N_move = N
        else          
            # adapt the next kernel
            _, curr_K, mask_idx = choose_weights_dfo(post, curr_K.recipe, θ, X, N_keep, k)
            pas .= min.(pas, pa_length(curr_K))
            all_K = ProductKernel(all_K, curr_K)
            @show pas

            # Put the kept particles at the end.
            keep_idx .= first(findall(isfinite, mask_idx), N_keep)

            resample_idx = keep_idx[vcat(rand(1:N_keep, N_drop), 1:N_keep)]
            θ .= θ[:, resample_idx]
            X .= X[:, resample_idx]
            ρ .= ρ[resample_idx]
        end

        q_cov .= cov(θ[:, 1:N_move], dims=2)
        local accs = zeros(Int, N_move)
        let R=R, all_K=all_K
            Threads.@threads for I = 1:N_move
                shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves = mcmc_sampler(post, all_K, R, MvNormal(q_cov); θ₀=θ[:, I], X₀=X[:, I])

                θ[:, I] .= shuffle_θ[:, end]
                X[:, I] .= shuffle_X[:, end]
                ρ[I] = last(shuffle_ρ)
                accs[I] = shuffle_moves
            end
        end
        final_iter && break
        p_acc = sum(accs)/(R*N_move)
        R = ceil(Int, log(1-p_acc, c))
        @show (p_acc, R)

        @show mean(θ, dims=2)
        @show var(θ, dims=2)

        final_iter = p_acc < p_thresh
    end

    return (θ, X, ρ, all_K)
end

## Non-uniform kernels
# smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe, N::Integer; drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
