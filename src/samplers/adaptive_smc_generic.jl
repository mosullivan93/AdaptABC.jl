# smc_state:
#  particles:
#   θ - parameters
#   X - simulations
#   ρ - distances
#   w - weights
#  kernel
# other options
#  R
#
#

# todo: How to save this object? I think it needs a custom serialisation method. Check the JLD2 docs.

struct AdaptiveSMCABCState{PosteriorType, KernelType, KernelAdaptationStrategyType, PerturbationStrategyType}
    # Properties of the particle population
    particles::Vector{Particle}
    summary_scales::Vector{Float64}
    # Properties of the (adaptive) kernel/posterior
    kernel::KernelType
    kas::KernelAdaptationStrategyType
    # Properties of the MCMC perturbation kernel
    ps::PerturbationStrategyType
    R::Int
    max_store_iters::Int
    # For record keeping
    cumsims::Int
    p_acc::Float64
    # Stopping condition
    p_thresh::Float64
end
parts(s::AdaptiveSMCABCState) = s.particles
kas(::AdaptiveSMCABCState{P, K, KAS, PS} where {P, K, PS}) where {KAS} = KAS

## should be two parts: 1) weight selection, 2) bandwidth determination
abstract type KernelAdaptationStrategy; end
abstract type AdaptiveWeighting <: KernelAdaptationStrategy; end
abstract type OptimalAdaptiveScaling <: AdaptiveWeighting; end
struct BandwidthOnlyAdaptation <: KernelAdaptationStrategy; target_ess::Int; end
struct ScaleReciprocal <: AdaptiveWeighting; target_ess::Int; end
struct OneAtATime <: OptimalAdaptiveScaling; target_ess::Int; end
struct ManifoldQuasiNewton <: OptimalAdaptiveScaling; target_ess::Int; end
struct ManifoldGradientDescent <: OptimalAdaptiveScaling; target_ess::Int; end
struct ManifoldNelderMead <: OptimalAdaptiveScaling; target_ess::Int; end
struct ManifoldDifferentialEvolution <: OptimalAdaptiveScaling; target_ess::Int; end

# is there a type that supports always returning the same thing for an index?
# responsible for two parts: 1) computing covs, 2) computing R
abstract type ParticlePerturbationStrategy; end
struct GlobalCovariance <: ParticlePerturbationStrategy; p_nomove::Float64; end
struct LocalCovariance <: ParticlePerturbationStrategy; p_nomove::Float64; end

function stablerowmad(x::Matrix{Float64})
    nr = size(x, 1)
    local mads = Vector{Float64}(undef, nr)
    for I = 1:nr
        local rm = median(x[I, :])
        mads[I] = median(abs.(x[I, :] .- rm))
    end
    return mads
end

function init(post::PosteriorType, N::Int, K::KernelType, kas::KernelAdaptationStrategy, pss::ParticlePerturbationStrategy, R₀, p_thresh) where {PosteriorType <: ImplicitPosterior, DistanceType, KernelType <: KernelRecipe{Uniform, DistanceType}}
    local kernel::Kernel{Uniform, DistanceType, ScalingTransform{1, length(post)}, PosteriorType}
    particles, kernel = rejection_sampler(post, K, N)
    pkernel = ProductKernel(kernel)
    summ_scales = stablerowmad(summ(particles))

    return AdaptiveSMCABCState{PosteriorType, typeof(pkernel), typeof(kas), typeof(pss)}(particles, summ_scales, pkernel, kas, pss, R₀, ifelse(kas isa AdaptiveWeighting, N*R₀, 1), N, 1.0, p_thresh)
end

find_opt_weights(s::AdaptiveSMCABCState, post::ImplicitPosterior, ake::AdaptiveKernelEstimator) = find_opt_weights(kas(s), s, post, ake)
function find_opt_weights(::Type{OneAtATime}, s::AdaptiveSMCABCState, post::ImplicitPosterior, ake::AdaptiveKernelEstimator)
    S = length(post)
    return argmin(Base.Fix1(logestb, ake), convert(Vector{Float64}, 1:S .== i) for i = 1:S)
end
function find_opt_weights(::Type{ManifoldDifferentialEvolution}, s::AdaptiveSMCABCState, post::ImplicitPosterior, ake::AdaptiveKernelEstimator)
    man = Manifolds.ProbabilitySimplex(Val(length(post)-1))

    man_obj(p) = logestb(ake, p./ s.summary_scales)
    opt_res = manifold_diffevo(man, man_obj, 500, max(length(post), 50))

    return opt_res./s.summary_scales
end
function find_opt_weights(::Type{ManifoldNelderMead}, s::AdaptiveSMCABCState, post::ImplicitPosterior, ake::AdaptiveKernelEstimator)
    man = Manifolds.ProbabilitySimplex(Val(length(post)-1))

    man_obj(::AbstractManifold, p) = logestb(ake, p ./ s.summary_scales)
    opt_res = Manopt.NelderMead(man, man_obj)

    return opt_res./s.summary_scales
end
function find_opt_weights(::Type{ManifoldQuasiNewton}, s::AdaptiveSMCABCState, post::ImplicitPosterior, ake::AdaptiveKernelEstimator)
    man = Manifolds.ProbabilitySimplex(Val(length(post)-1))

    obj(p) = logestb(ake, p ./ s.summary_scales)
    man_obj(::AbstractManifold, p) = obj(p)
    g_man_obj(::AbstractManifold, p) = ManifoldDiff.gradient(man, obj, p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))

    n_multistarts = 50
    st_pts = rand(man, n_multistarts)
    ms_res = Vector{Tuple{Float64, Vector{Float64}}}(undef, n_multistarts)
    Threads.@threads for I = 1:n_multistarts
        opt_res = Manopt.quasi_Newton(man, man_obj, g_man_obj, st_pts[I], debug=[:Iteration,(:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), "\n", :Stop, 25])
        ms_res[I] = (obj(opt_res), opt_res)
    end
    opt_res = last(argmin(first, ms_res))

    return opt_res./s.summary_scales
end
function find_opt_weights(::Type{ManifoldGradientDescent}, s::AdaptiveSMCABCState, post::ImplicitPosterior, ake::AdaptiveKernelEstimator)
    man = Manifolds.ProbabilitySimplex(Val(length(post)-1))

    obj(p) = logestb(ake, p ./ s.summary_scales)
    man_obj(::AbstractManifold, p) = obj(p)
    g_man_obj(::AbstractManifold, p) = ManifoldDiff.gradient(man, obj, p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))

    n_multistarts = 50
    st_pts = vcat([ones(length(post))/length(post)], rand(man, n_multistarts-1))
    ms_res = Vector{Tuple{Float64, Vector{Float64}}}(undef, n_multistarts)
    Threads.@threads for I = 1:n_multistarts
        opt_res = Manopt.gradient_descent(man, man_obj, g_man_obj, st_pts[I])
        ms_res[I] = (obj(opt_res), opt_res)
    end
    opt_res = last(argmin(first, ms_res))

    return opt_res./s.summary_scales
end

# Use an adaptive scaling transform according to the adaptation strategy's weighting optimiser
choose_weights(s::AdaptiveSMCABCState{P, K, ScaleReciprocal, PS} where {P, K, PS}, ::ImplicitPosterior) = 1.0 ./ s.summary_scales
function choose_weights(s::AdaptiveSMCABCState, post::ImplicitPosterior)
    ake = AdaptiveKernelEstimator(ifelse(_density(s.kernel) isa Uniform, SubsetSampleBC, WeightedSampleBC), post, param(s.particles), summ(s.particles), recipe(s.kernel), s.kas.target_ess)
    return find_opt_weights(s, post, ake)
end


# Reuse the existing kernel recipe to preserve a potential initial custom ScalingTransform
next_kernel_refinement(s::AdaptiveSMCABCState{P, K, BandwidthOnlyAdaptation, PS} where {P, K, PS}, post::ImplicitPosterior) = adapt_ess(post, recipe(s.kernel), s.particles, s.kas.target_ess)

# Otherwise, let the adaptation strategy determine the weights
function next_kernel_refinement(s::AdaptiveSMCABCState, post::ImplicitPosterior)
    weights = choose_weights(s, post)
    newk = revise(recipe(s.kernel), t=ScalingTransform(weights, Val(length(post))))
    return adapt_ess(post, newk, s.particles, s.kas.target_ess)
end

# Refinement for next iteration involves finding the next kernel component and determining which particles remain and how the others are replaced
function next_refinement(s::AdaptiveSMCABCState, post::ImplicitPosterior)
    newK = next_kernel_refinement(s, post)
    keep_idx, move_idx = particle_propagation_plan(newK, s)
    return ProductKernel(s.kernel, newK), keep_idx, move_idx
end

particle_propagation_plan(k::PosteriorApproximator{Uniform}, s::AdaptiveSMCABCState) = particle_propagation_plan(isfinite.(logpdfu.(k, summ.(s.particles))), s.kas.target_ess)
particle_propagation_plan(k::PosteriorApproximator, s::AdaptiveSMCABCState) = particle_propagation_plan(pdfu.(k, summ.(s.particles)))
# convert a bitmask or weights into a list of particles to keep and the starting positions of particles being replaced/jittered
particle_propagation_plan(kernel_output) = error("something wrong")
particle_propagation_plan(kernel_output::Vector{Float64}) = error("todo: implement the weighted resampling strategy")
function particle_propagation_plan(kernel_output::BitVector, N_keep::Int)
    keep_idx = first(findall(kernel_output), N_keep)
    move_idx = rand(keep_idx, length(kernel_output) - length(keep_idx))
    return keep_idx, move_idx
end

compute_mcmc_covs(s::AdaptiveSMCABCState{P, K, KA, GlobalCovariance} where {P, K, KA}, keep_idx, move_idx) = fill(cov(param(s.particles[keep_idx]), dims=2), length(move_idx))
function compute_mcmc_covs(s::AdaptiveSMCABCState{P, K, KA, LocalCovariance} where {P, K, KA}, keep_idx, move_idx)
    kept_θ = param(s.particles[keep_idx])
    kept_N = length(keep_idx)
    covs = Vector{Matrix{Float64}}(undef, maximum(move_idx))
    for pind in unique(move_idx)
        local v::Matrix{Float64} = kept_θ .- param(s.particles[pind])
        covs[pind] = v*v'/kept_N
    end
    return covs[move_idx]
end

function final_perturbation(s::AdaptiveSMCABCState, post::ImplicitPosterior)
    # Compute MCMC tuning parameters using all parameters
    covs = compute_mcmc_covs(s, 1:length(s.particles), 1:length(s.particles))
    # Move all particles in the final perturbation
    parts, scales, N_sims, acc, R = perturb_particles(s, post, s.kernel, covs, deepcopy(s.particles))

    # return state
    return AdaptiveSMCABCState{typeof(post), typeof(s.kernel), typeof(s.kas), typeof(s.ps)}(parts, scales, s.kernel, s.kas, s.ps, R, s.max_store_iters, s.cumsims + N_sims, acc, s.p_thresh)
end

function perturb_particles(s::AdaptiveSMCABCState, post::ImplicitPosterior, K::PosteriorApproximator, covs::Vector{Matrix{Float64}}, parts::Vector{Particle})
    local N_move = length(covs)
    local accs = zeros(Int, N_move)
    local n_sims_stored = min(s.max_store_iters, s.R*N_move)
    local all_sims = zeros(Float64, length(post), n_sims_stored)
    local store_sims = Threads.Atomic{Bool}(s.kas isa AdaptiveWeighting)
    local saved_sims = Threads.Atomic{Int}(1)

    Threads.@threads for I = 1:N_move
        shuffle_parts, shuffle_moves, shuffle_uniq_sims = mcmc_sampler(post, K, s.R, MvNormal(covs[I]), parts[I], store_uniq_sims=store_sims[])

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

        parts[I] = last(shuffle_parts)
        accs[I] = shuffle_moves
    end
    last_sim = ifelse(s.kas isa AdaptiveWeighting, min(n_sims_stored, saved_sims[] - 1), 1)
    scales = stablerowmad(all_sims[:, 1:last_sim])
    N_sims = N_move*s.R
    @show p_acc = sum(accs)/N_sims
    @show next_R = ceil(Int, log(1-p_acc, s.ps.p_nomove))

    return parts, scales, N_sims, p_acc, next_R
end

function particle_replenishment(s::AdaptiveSMCABCState, post::ImplicitPosterior, K::PosteriorApproximator, keep_idx::Vector{Int64}, move_idx::Vector{Int64})
    # Persist kept particles at the end, and initialise replenished particles at the beginning
    pts = deepcopy(s.particles[vcat(move_idx, keep_idx)])

    # Compute MCMC tuning parameters
    covs = compute_mcmc_covs(s, keep_idx, move_idx)
    # Length of covs will determine how many particles are moved
    return perturb_particles(s, post, K, covs, pts)
end

function iteratesmc(s::AdaptiveSMCABCState, post::ImplicitPosterior)
    # get kernel using state
    kernel, keep_idx, move_idx = next_refinement(s, post)

    # perform filtration/replenishment using mcmc
    parts, scales, N_sims, acc, R = particle_replenishment(s, post, kernel, keep_idx, move_idx)

    # return state
    return AdaptiveSMCABCState{typeof(post), typeof(s.kernel), typeof(s.kas), typeof(s.ps)}(parts, scales, kernel, s.kas, s.ps, R, s.max_store_iters, s.cumsims + N_sims, acc, s.p_thresh)
end

isdone(s::AdaptiveSMCABCState) = s.p_acc < s.p_thresh

# function adaptive_smc_sampler_opt_dfo(post::ImplicitPosterior{M, P, S}, K::KernelRecipe{Uniform, D, T}, N::Int, drop::Percentage=%(50), R₀::Int=10, p_thresh::Float64=0.05, c::Float64=0.05, k::Union{Int, Missing}=missing) where {M, P, S, D, T}
function adaptive_smc_generic(post::ImplicitPosterior, N::Int, K::KernelRecipe{Uniform}, ::Type{kas}, ::Type{pss}, drop::Percentage=%(50), R₀::Int=10, p_thresh::Float64=0.05, c::Float64=0.05) where {kas<:KernelAdaptationStrategy, pss <: ParticlePerturbationStrategy}
    # initialise algorithm state
    state = init(post, N, K, kas(N-drop(N)), pss(c), R₀, p_thresh)
    # Can initialise this with an empty vector so that history stays as a cons type to ensure stability. vcat will ignore it.
    history = cons([state], nil(Vector{typeof(state)}))
    # iterate state
    while !isdone(state)
        state = iteratesmc(state, post)
        history = cons([state], history)
    end

    state = final_perturbation(state, post)
    history = cons([state], history)
    # do final perturbation?

    # return something
    parts(state), reduce(vcat, reverse(history))
end