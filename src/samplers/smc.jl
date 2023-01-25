# function smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe{Uniform}, N::Integer;
#     drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}

# function smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe{Uniform}, N::Integer,
#     drop_prop::Float64=0.5, R₀=10, p_thresh=0.05, c=0.05) where {M}

# function smc_sampler(post::ImplicitPosterior{M, P, S}, K::KernelRecipe{Uniform, D, T}, N::Integer,
#                      drop_prop::Float64, R₀::Integer, p_thresh::Float64, c::Float64) where {M, P, S, D, T}
function smc_sampler(post::ImplicitPosterior{M, P, S}, K::KernelRecipe{Uniform, D, T}, N::Integer;
    drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M, P, S, D, T}
    # Setup
    π = prior(post)

    # Setup consts
    local N_drop::Int = drop(N)
    local N_keep::Int = N - N_drop
    local N_move::Int = N_drop
    local R::Int = R₀
    
    # Initialise particles
    local θ::Matrix{Float64}
    local oθ::Matrix{Float64}
    local X::Matrix{Float64}
    local ρ::Vector{Float64}
    local curr_K::Kernel{Uniform, D, T, ImplicitPosterior{M, P, S}}

    #* always sorted by this method
    θ, X, ρ, curr_K = rejection_sampler(post, K, N)
    @show curr_K.bandwidth
    oθ = similar(θ)

    local q_cov::Matrix{Float64} = Matrix{Float64}(undef, length(π), length(π))
    local p_acc::Float64 = 1.0
    final_iter = false
    while true
        if final_iter
            N_move = N
        else
            #* Put kept samples at end and iterate 1:N_drop
            resample_idx = vcat(rand(1:N_keep, N_drop), 1:N_keep)
            θ .= θ[:, resample_idx]
            X .= X[:, resample_idx]
            ρ .= ρ[resample_idx]

            curr_K = revise(curr_K, ϵ=last(ρ))
            @show curr_K.bandwidth
        end

        q_cov .= cov(θ[:, 1:N_move], dims=2)
        local accs = zeros(Int, N_move)
        let R=R, curr_K=curr_K
            Threads.@threads for I = 1:N_move
                shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves = mcmc_sampler(post, curr_K, R, MvNormal(q_cov); θ₀=θ[:, I], X₀=X[:, I])

                θ[:, I] .= shuffle_θ[:, end]
                X[:, I] .= shuffle_X[:, end]
                ρ[I] = last(shuffle_ρ)
                accs[I] = shuffle_moves
            end
        end
        final_iter && break
        p_acc = sum(accs)/(R*N_drop)
        R = ceil(Int, log(1-p_acc, c))
        @show p_acc, R

        idx = sortperm(ρ)
        θ .= θ[:, idx]
        X .= X[:, idx]
        ρ .= ρ[idx]

        if π isa Bijectors.TransformedDistribution
            oθ .= invlink(post.bayesmodel, θ)
            @show mean(oθ, dims=2)
            @show var(oθ, dims=2)
        else
            @show mean(θ, dims=2)
            @show var(θ, dims=2)
        end

        final_iter = p_acc < p_thresh
    end

    return (θ, X, ρ, curr_K)
end

## Non-uniform kernels
# smc_sampler(mdl::ImplicitPosterior{M}, K::KernelRecipe, N::Integer; drop::Percentage=%(50), R₀=10, p_thresh=0.05, c=0.05) where {M}
