# # ? Using Manopt's DFO methods - NM is good, PSO is very slow but can work better with enough particles.
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     # M = Manifolds.Sphere(Val(S - 1))
#     M = Manifolds.ProbabilitySimplex(Val(S-1))
#     rp = randperm(S)
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     # adaptive_estimator = AdaptiveKernelEstimator(WeightedSampleBC, post, th, xs, K, N_keep; k=k)
#     scales = vec(std(xs, dims=2))

#     man_obj(::AbstractManifold, p) = first(logestb(adaptive_estimator, ScalingTransform(p[rp]./scales, Val(S))))
#     # mopt_res = Manopt.NelderMead(M, man_obj, NelderMeadSimplex(collect(eachrow(diagm(ones(S))))))
#     mopt_res = Manopt.NelderMead(M, man_obj)
#     # mopt_res = particle_swarm(M, man_obj, n=500)
#     # mopt_res = particle_swarm(M, man_obj)
#     return logestb(adaptive_estimator, ScalingTransform(mopt_res[rp]./scales, Val(S)))
#     # return logestb(adaptive_estimator, ScalingTransform(abs.(mopt_res)./scales, Val(S)))
# end

# ? Spherical Surrogate + Manifold PSO
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     # This would **HEAVILY** depend on the quality of the surrogate... maybe need a lot more points.
#     surr = build_spherical_surrogate(post, th, xs, adaptive_estimator; n_pts=S*100)
#     # surr = build_spherical_surrogate(post, th, xs, adaptive_estimator; n_pts=S*100)
#     # adaptive_estimator = AdaptiveKernelEstimator(WeightedSampleBC, post, th, xs, K, N_keep; k=k)
#     scales = vec(std(xs, dims=2))

#     man_obj(::AbstractManifold, p) = surr(p)
#     mopt_res = Manopt.NelderMead(Sphere(Val(S-1)), man_obj, NelderMeadSimplex(collect(eachrow(diagm(ones(S))))))
#     # mopt_res = Manopt.NelderMead(Sphere(Val(S-1)), man_obj)
#     # mopt_res = particle_swarm(Sphere(Val(S-1)), man_obj, n=500)
#     # mopt_res = particle_swarm(Sphere(Val(S-1)), man_obj)
#     return logestb(adaptive_estimator, ScalingTransform(abs.(mopt_res)./scales, Val(S)))
# end

#? Custom Manifold Differential Evolution.
function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
    # man = Manifolds.Sphere(Val(S-1))
    man = Manifolds.ProbabilitySimplex(Val(S-1))
    rp = randperm(S)
    adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
    scales = vec(std(xs, dims=2))
    
    man_obj(p) = first(logestb(adaptive_estimator, ScalingTransform(p[rp]./scales, Val(S))))
    # opt_res = manifold_diffevo(man, man_obj, 200, 50)
    opt_res = manifold_diffevo(man, man_obj, 500, 100)
    # opt_res = manifold_diffevo(man, man_obj, 100, 100)
    return logestb(adaptive_estimator, ScalingTransform(opt_res[rp]./scales, Val(S)))

    # man_obj(p) = first(logestb(adaptive_estimator, ScalingTransform(p[rp]./scales)))
    # opt_res = abs.(manifold_diffevo(man, man_obj, 100, 100))
    # opt_res = abs.(manifold_diffevo(man, man_obj; niters=1000, n_pop=S*20))
    # opt_res = abs.(manifold_diffevo(man, man_obj; pt_per_dim=10))
    # return logestb(adaptive_estimator, ScalingTransform(opt_res[rp]./scales, Val(S)))
end

# #? Using my bayesion optimisation on a sphere method.
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     surr = build_spherical_surrogate(post, th, xs, adaptive_estimator; n_pts=S+100)
#     # surr = build_spherical_surrogate_bs(post, th, xs, adaptive_estimator; n_pts=S*20)
#     scales = vec(std(xs, dims=2))
#     opt_res = manifold_surrogate_opt_hybrid(Manifolds.Sphere(Val(S-1)), p -> first(logestb(adaptive_estimator, ScalingTransform(p./scales, Val(length(post))))), surr)
#     return logestb(adaptive_estimator, ScalingTransform(opt_res./scales, Val(S)))
# end

# #? Using my bayesion optimisation on a sphere method.
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     # surr = build_spherical_surrogate(post, th, xs, adaptive_estimator; n_pts=S+100)
#     surr = build_spherical_surrogate_bs(post, th, xs, adaptive_estimator; n_pts=S*20)
#     scales = vec(std(xs, dims=2))
#     # manifold_surrogate_opt_direct(Manifolds.Sphere(Val(S-1)), p -> first(logestb(adaptive_estimator, ScalingTransform(p./scales, Val(length(post))))), surr; maxiters=1)
#     manifold_surrogate_opt_ei(Manifolds.Sphere(Val(S-1)), p -> first(logestb(adaptive_estimator, ScalingTransform(p./scales, Val(length(post))))), surr)
#     # surrsphereopt(p -> first(logestb(adaptive_estimator, ScalingTransform(p./scales, Val(length(post))))), surr)
#     # bayessphereopt(p -> first(logestb(adaptive_estimator, ScalingTransform(p./scales, Val(length(post))))), surr)

#     @show surr.x[argmin(surr.y)]
#     return logestb(adaptive_estimator, ScalingTransform(surr.x[argmin(surr.y)]./scales, Val(S)))
# end

#? SphericalSurrogate + BlackBoxOptim
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     # This would **HEAVILY** depend on the quality of the surrogate... maybe need a lot more points.
#     surr = build_spherical_surrogate(post, th, xs, adaptive_estimator; n_pts=S*100)
#     # adaptive_estimator = AdaptiveKernelEstimator(WeightedSampleBC, post, th, xs, K, N_keep; k=k)
#     scales = vec(std(xs, dims=2))

#     obj(ϕ) = surr(Ω(ϕ))
#     bbres = bboptimize(obj;
#                     SearchRange = fill((0.0, pi/2), S-1),
#                     NumDimensions = S-1,
#                     NThreads = Threads.nthreads()-1,
#                     Method = :adaptive_de_rand_1_bin_radiuslimited,
#                     # Method = :dxnes,
#                     TraceMode = :silent
#                 )
#     res = Ω(best_candidate(bbres))
#     return logestb(adaptive_estimator, ScalingTransform(res./scales, Val(S)))
# end

#? Using BlackBoxOptim - Best so far, parallelised.
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     # adaptive_estimator = AdaptiveKernelEstimator(WeightedSampleBC, post, th, xs, K, N_keep; k=k)
#     scales = vec(std(xs, dims=2))

#     obj(ϕ) = first(logestb(adaptive_estimator, ScalingTransform(Ω(ϕ)./scales)))
#     # bbres = bboptimize(obj, Ω⁻¹(ones(S));
#     bbres = bboptimize(obj;
#                     SearchRange = fill((0.0, pi/2), S-1),
#                     NumDimensions = S-1,
#                     NThreads = Threads.nthreads()-1,
#                     Method = :adaptive_de_rand_1_bin_radiuslimited,
#                     # MaxSteps = max(1e4, S*1000),
#                     # Method = :dxnes,
#                     # PopulationSize = 1000,
#                     TraceMode = :silent
#                 )
#     res = Ω(best_candidate(bbres))
#     return logestb(adaptive_estimator, ScalingTransform(res./scales, Val(S)))
# end

#? Using Optim.jl - These are good, but struggle with the noise.
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     # adaptive_estimator = AdaptiveKernelEstimator(WeightedSampleBC, post, th, xs, K, N_keep; k=k)
#     scales = vec(std(xs, dims=2))

#     obj(ϕ) = first(logestb(adaptive_estimator, ScalingTransform(Ω(ϕ)./scales)))
#     # res = Optim.optimize(obj, fill(0.0, S-1), fill(pi/2, S-1), Optim.ParticleSwarm(lower=fill(0.0, S-1), upper=fill(pi/2, S-1), n_particles=100))
#     # res = Optim.optimize(obj, fill(0.0, S-1), fill(pi/2, S-1), Optim.NelderMead())
#     res = Optim.optimize(obj, fill(0.0, S-1), fill(pi/2, S-1), Ω⁻¹(ones(S)), SAMIN(), Optim.Options(iterations=10^6))

#     res = Ω(Optim.minimizer(res))
#     return logestb(adaptive_estimator, ScalingTransform(res./scales, Val(S)))
# end

#? Using Surrogates.jl - Works well. I'ts very fast. Still need to compare the quality of the answer to BlackBoxOptim. (better on the toads)
# function choose_weights_dfo(post::ImplicitPosterior{M, P, S} where {M, P}, K::KernelRecipe{Uniform}, th::AbstractMatrix, xs::AbstractMatrix, N_keep::Integer, k::Union{Int, Missing}=missing) where {S}
#     adaptive_estimator = AdaptiveKernelEstimator(SubsetSampleBC, post, th, xs, K, N_keep; k=k)
#     # adaptive_estimator = AdaptiveKernelEstimator(WeightedSampleBC, post, th, xs, K, N_keep; k=k)
#     scales = vec(std(xs, dims=2))
#     obj(ϕ) = first(logestb(adaptive_estimator, ScalingTransform(Ω(ϕ)./scales)))

#     # Construct surrogate
#     #  fdg(n) = digits.(Bool, 1:2^n-1, base=2, pad=n)
#     # also, can do stratified sampling by taking points on the sphere, and converting with Omega-Inv, then replacing the last angle (won't affect density).
#     #! todo: Fix DYCORS so that it adds points regardless of if they're better...
#     #! Otherwise, SRBF is better for now. Kriging is too slow with SRBF.
#     #! todo: Make a loop that generates new surrogates with random points and tries again.
#     #! todo: In aforementioned loop, consider tempering the generalised normal?

#     n_pts = 100
#     # pts = Tuple.(Ω⁻¹.(map.(abs, rand(Manifolds.Sphere(Val(S - 1)), n_pts - S - 1))))
#     pts = Tuple.(Ω⁻¹.(map.(abs, vcat(rand(Manifolds.Sphere(Val(S - 1)), n_pts - S - 1), collect(eachrow(diagm(ones(S)))), [ones(S)]))))
#     # pts = Tuple.(Ω⁻¹.(vcat(collect.(stratified_sphere_sample(S, n_pts - S - 1)), collect(eachrow(diagm(ones(S)))), [ones(S)])))
#     # pts = vcat(stratified_sphere_sample(S, n_pts - S - 1), collect(eachrow(diagm(ones(S)))), [ones(S)])))
#     # pts = stratified_sphere_sample(S, n_pts)
#     vals = obj.(pts)

#     # surr = LobachevskySurrogate(pts, vals, fill(0.0, S-1), fill(pi/2, S-1))
#     # surr = Kriging(pts, vals, fill(0.0, S-1), fill(pi/2, S-1))
#     surr = RadialBasis(pts, vals, fill(0.0, S-1), fill(pi/2, S-1))
#     # n_iters = max(200, 50S)
#     n_iters = 100
#     # @show res = Ω(first(surrogate_optimize(obj, SRBF(), fill(0.0, S-1), fill(pi/2, S-1), surr, SobolSample())))
#     # @show res = Ω(first(surrogate_optimize(obj, SRBF(), fill(0.0, S-1), fill(pi/2, S-1), surr, UniformSample())))
#     # @show res = Ω(first(surrogate_optimize(obj, DYCORS(), fill(0.0, S-1), fill(pi/2, S-1), surr, SobolSample(), maxiters=10000)))
#     @show res = Ω(first(surrogate_optimize(obj, DYCORS(), fill(0.0, S-1), fill(pi/2, S-1), surr, LatinHypercubeSample(), maxiters=n_iters)))
#     # @show res = Ω(first(surrogate_optimize(obj, DYCORS(), fill(0.0, S-1), fill(pi/2, S-1), surr, SobolSample(), maxiters=n_iters)))
#     # @show res = Ω(first(surrogate_optimize(obj, DYCORS(), fill(0.0, S-1), fill(pi/2, S-1), surr, UniformSample())))

#     # surrogate_optimize(obj, SRBF(), fill(0.0, S-1), fill(pi/2, S-1), surr, LatinHypercubeSample(), maxiters=n_iters)
#     # sphereopt(obj, EI(), surr, maxiters=n_iters, num_new_samples=16)
#     # bayesopt(obj, EI(), fill(0.0, S-1), fill(pi/2, S-1), surr, StratifiedSphericalSampler{S}(), maxiters=n_iters, num_new_samples=16)
#     # surrogate_optimize(obj, EI(), fill(0.0, S-1), fill(pi/2, S-1), surr, StratifiedSphericalSampler{S}(), maxiters=n_iters)
#     # surrogate_optimize(obj, EI(), fill(0.0, S-1), fill(pi/2, S-1), surr, LatinHypercubeSample(), maxiters=n_iters)
#     # surrogate_optimize(obj, LCBS(), fill(0.0, S-1), fill(pi/2, S-1), surr, LatinHypercubeSample(), maxiters=n_iters)
#     @show res = Ω(surr.x[argmin(surr.y)])

#     return logestb(adaptive_estimator, ScalingTransform(res./scales, Val(S)))
# end

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
    local curr_K::Kernel{Uniform, D, ScalingTransform{1, S}, typeof(post)}
    local all_K::ProductKernel{Uniform, D, ScalingTransform{1, S}, typeof(post)}
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

    # #? INSERT TEST
    # #! Maybe I want to be performing the first weight optimisation after the particles have had a chance to undergo MCMC. This way it can move towards the high density regions.
    # q_cov .= cov(θ, dims=2)/2
    # let R=R, all_K=all_K
    #     Threads.@threads for I = 1:N
    #         shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves, shuffle_bnds = mcmc_sampler(post, all_K, R, MvNormal(q_cov); θ₀=θ[:, I], X₀=X[:, I])

    #         θ[:, I] .= shuffle_θ[:, end]
    #         X[:, I] .= shuffle_X[:, end]
    #         ρ[I] = last(shuffle_ρ)
    #     end
    # end
    # #? END TEST

    covs::Vector{Matrix{Float64}} = Matrix{Float64}[]
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
            kept_θ = θ[:, keep_idx]
            # covs = map(v -> v*v', kept_θ .- c for c in eachcol(kept_θ))

            resample_idx = keep_idx[vcat(rand(1:N_keep, N_drop), 1:N_keep)]
            θ .= θ[:, resample_idx]
            X .= X[:, resample_idx]
            ρ .= ρ[resample_idx]

            covs = map(v -> v*v', kept_θ .- c for c in eachcol(θ))/N_keep
        end

        # covs = [cov(θ[:, i[1:N_keep]], dims=2) for i in sortperm.(eachcol(pairwise(euclidean, X)))]
        # q_cov .= cov(θ[:, end-N_move+1:end], dims=2)/2
        local accs = zeros(Int, N_move)
        local oobs = zeros(Int, N_move)
        let R=R, all_K=all_K, covs=covs
        # let R=R, all_K=all_K
            Threads.@threads for I = 1:N_move
            # for I = 1:N_move
                shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves, shuffle_bnds = mcmc_sampler(post, all_K, R, MvNormal(covs[I]/2); θ₀=θ[:, I], X₀=X[:, I])
                # shuffle_θ, shuffle_X, shuffle_ρ, shuffle_moves, shuffle_bnds = mcmc_sampler(post, all_K, R, MvNormal(q_cov); θ₀=θ[:, I], X₀=X[:, I])

                θ[:, I] .= shuffle_θ[:, end]
                X[:, I] .= shuffle_X[:, end]
                ρ[I] = last(shuffle_ρ)
                accs[I] = shuffle_moves
                # oobs[I] = shuffle_bnds
            end
        end
        final_iter && break
        @show sum(oobs)/(R*N_move)
        @show p_acc = sum(accs)/(R*N_move)
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
