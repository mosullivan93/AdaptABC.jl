# Distance metric that measures the geodesic on the surface of a sphere.
struct ManifoldDistance{M<:AbstractManifold} <: PreMetric
    manifold::M
end

function Distances._evaluate(d::ManifoldDistance, a, b)
    # @boundscheck if !is_point(d.manifold, a) || !is_point(d.manifold, b)
    #     throw(DimensionMismatch("Either a ($a) or b ($b) is not on the manifold $(d.manifold)."))
    # end
    # @boundscheck if !(norm(a) ≈ 1.0 && norm(b) ≈ 1.0)
    #     throw(DimensionMismatch("Either a ($a) or b ($b) is not on the manifold $(d.manifold)."))
    # end
    #! try projecting a/b onto manifold first?
    return Manifolds.distance(d.manifold, a, b)
end
# result_type is required for a colwise (or pairwise, I forget) call to estimate the std_error. Should check how to use it when doing more complicated manifolds.
@inline Distances.result_type(::ManifoldDistance, a, b) = Float64
@inline (dist::ManifoldDistance)(a, b) = Distances._evaluate(dist, a, b)

# Quasi random sampling on a sphere that stratifies over the azimuthal dimension to ensure good representation.
function stratified_sphere_sample(M::Sphere, n_pts::Int)
    # Preallocate and sample from M (required for type-stability)
    sph_samples = [allocate_result(M, rand) for _ in 1:n_pts]
    rand!.(Ref(M), sph_samples);
    # Restrict points to positive orthant.
    pts = mapreduce(Ω⁻¹, hcat, sph_samples)
    # Stratify on slices in the azimuthal angle.
    az = 2π*(rand(n_pts)/n_pts .+ range(0, step=1/n_pts, length=n_pts))
    # Replace coordinate
    pts[end, :] .= az

    return Ω.(eachcol(pts))
end

# Quasi random sampling on a sphere that stratifies over the azimuthal dimension to ensure good representation.
function stratified_possphere_sample(M::Sphere, n_pts::Int)
    # Preallocate and sample from M (required for type-stability)
    sph_samples = [allocate_result(M, rand) for _ in 1:n_pts]
    rand!.(Ref(M), sph_samples);
    # Restrict points to positive orthant.
    pts = mapreduce(Ω⁻¹, hcat, map.(abs, sph_samples))
    # Stratify on slices in the azimuthal angle.
    az = π/2*(rand(n_pts)/n_pts .+ range(0, step=1/n_pts, length=n_pts))
    # Replace coordinate
    pts[end, :] .= az

    return Ω.(eachcol(pts))
end

# n is the maximal bit position, k is the maximal bits to set, start_pos is where we're at
perms(n::Int, k::Int=n) = BitVector.(digits.(Bool, _perms(n, k), base=2, pad=n))

function _perms(n::Int, k::Int=n, start_pos::Int=0, start_byte::UInt128=UInt128(0))
    if (k == 0) || (n == start_pos)
        (start_byte > 0) && return UInt128[start_byte]
        return []
    end

    result = UInt128[]
    append!(result, _perms(n, k-1, start_pos+1, start_byte | (UInt128(1) << start_pos)))
    append!(result, _perms(n,   k, start_pos+1, start_byte))

    return result
end

# pts = reduce(partial(cat; dims=length(representation_size(S))+1), exp(S, p, t) for t in rand_tangent(S, p, 100));
# ds = [distance(S, p, pt) for pt in eachslice(pts, dims=ndims(pts))];

# todo: consider using retraction for the random tanget rather than exp.
function rand_tangent(man::AbstractManifold, pt, n=1; scale::Real=1.0, basis::AbstractBasis=DefaultOrthonormalBasis())
    dir_choice = Manifolds.Sphere(Val(manifold_dimension(man) - 1))
    # Preallocate and sample from M (required for type-stability)
    dirs = [allocate_result(dir_choice, rand) for _ in 1:n]
    rand!.(Ref(dir_choice), dirs);

    mag = abs.(randn(n)*scale)
    return [get_vector(man, pt, v, basis) for v in mag.*dirs]
end

# rand(Manifolds.Rotations(4))

# # The magnitude of the exponential controls the spread of the points. Can be set such that the mean distance is 1/2 to the neighbour?
# Can also use a half_normal/abs of normal to ensure we get MVN samples? Set such that 3stds is next closest point?

# Slightly optimised versions which update the posterior covariance matrix rather than recompute it.
function add_surr_points!(g::AbstractGPSurrogate, new_x::Vector{Vector{T}}, new_y::Vector{T}, eps=nothing) where {T}
    newpts = findall(!in(g.x), new_x)
    if length(newpts) < length(new_x)
        println("While appending to surrogate, $(length(new_x) - length(newpts))/$(length(new_x)) sample(s) were already present and have been ignored.")
        return
    end
    g.x = vcat(g.x, new_x)
    g.y = vcat(g.y, new_y)
    # Use the current posterior at the new points as the prior for adding the new observations.
    g.gp_posterior = posterior(g.gp_posterior(new_x, something(eps, g.Σy)), new_y)
    nothing
end
add_surr_point!(g::SurrogatesAbstractGPs.AbstractGPSurrogate, new_x, new_y, eps=nothing) = add_surr_points!(g, [new_x], [new_y], eps)

# function expected_decrease(x, surr, f_min, ξ) #maximise this
#     σ = std_error_at_point(surr, x)
#     f_prop = surr(x)
#     z = ((f_min - ξ) - f_prop)/σ

#     return σ*(z*cdf(Normal(), z) + pdf(Normal(), z))
# end
function expected_decrease(x, surr, f_min, ξ) #maximise this
    f_prop, σ² = only.(mean_and_var(surr.gp_posterior, [x]))
    σ = sqrt(σ²)
    δz = ((f_min - ξ) - f_prop)
    return δz*cdf(Normal(), δz/σ) + pdf(Normal(), δz/σ)
    # return log(σ) + log(z*cdf(Normal(), z) + pdf(Normal(), z))
end

function log_prob_decrease(x, surr, f_min) #maximise this
    f_prop, σ² = only.(mean_and_var(surr.gp_posterior, [x]))
    return logcdf(Normal(), (f_min - f_prop)/sqrt(σ²))
end
prob_decrease(x, surr, f_min) = exp(log_prob_decrease(x, surr, f_min))

function lower_conf_band(x, surr, λ=2) #minimise this
    f_μ, f_σ² = only.(mean_and_var(surr.gp_posterior, [x]))
    return f_μ - λ*sqrt(f_σ²)
end

function build_spherical_surrogate_bs(post::ImplicitPosterior{M, P, S} where {M, P},
                                   th_samples::Matrix{Float64}, xs_samples::Matrix{Float64},
                                   ake::AdaptiveKernelEstimator{K, M, E} where {K, M};
                                   n_pts = 100, n_bs = 5) where {S, E}

    M = Manifolds.Sphere(Val(S-1))
    # train_x =  map.(abs, stratified_possphere_sample(M, n_pts))
    train_x =  map.(abs, vcat(stratified_possphere_sample(M, n_pts - S - 1), convert.(Vector{Float64}, perms(S, 1)), [normalize(ones(S))]))
    train_y = Vector{Float64}(undef, n_pts)
    bs_train_y = Matrix{Float64}(undef, (n_pts, n_bs))

    scs = vec(std(xs_samples, dims=2))
    Threads.@threads for I = 1:n_pts
        train_y[I] = first(logestb(ake, ScalingTransform(train_x[I]./scs, Val(S))))
    end

    for B = 1:n_bs
        local bs_est = AdaptiveKernelEstimator(BootstrappedDistanceEstimator{E}, post, th_samples, xs_samples, ake.recipe, ake.target_ess; k=ake.estimator.k)
        Threads.@threads for I = 1:n_pts
            bs_train_y[I, B] = first(logestb(bs_est, ScalingTransform(train_x[I]./scs, Val(S))))
        end
    end

    map_pos_orth(p) = abs.(p)
    # return AbstractGPSurrogate(train_x, train_y, gp=GP(Matern52Kernel(metric=ManifoldDistance(M))), Σy=mean(var(bs_train_y, dims=2)))
    return AbstractGPSurrogate(train_x, train_y, gp=GP(Matern52Kernel(metric=ManifoldDistance(M)) ∘ FunctionTransform(map_pos_orth)), Σy=mean(var(bs_train_y, dims=2)))
    # return AbstractGPSurrogate(train_x, train_y, gp=GP(GaussianKernel(metric=ManifoldDistance(M)) ∘ FunctionTransform(map_pos_orth)), Σy=mean(var(bs_train_y, dims=2)))
end

function build_spherical_surrogate(post::ImplicitPosterior{M, P, S} where {M, P},
                                   th_samples::Matrix{Float64}, xs_samples::Matrix{Float64},
                                   ake::AdaptiveKernelEstimator{K, M, E} where {K, M};
                                   n_pts = 100) where {S, E}

    M = Manifolds.Sphere(Val(S-1))
    train_x =  map.(abs, vcat(stratified_possphere_sample(M, n_pts - S - 1), convert.(Vector{Float64}, perms(S, 1)), [normalize(ones(S))]))
    train_y = Vector{Float64}(undef, n_pts)

    scs = vec(std(xs_samples, dims=2))
    Threads.@threads for I = 1:n_pts
        train_y[I] = first(logestb(ake, ScalingTransform(train_x[I]./scs, Val(S))))
    end

    map_pos_orth(p) = abs.(p)
    # return AbstractGPSurrogate(train_x, train_y, gp=GP(Matern52Kernel(metric=ManifoldDistance(M))), Σy=1e-6)
    return AbstractGPSurrogate(train_x, train_y, gp=GP(Matern52Kernel(metric=ManifoldDistance(M)) ∘ FunctionTransform(map_pos_orth)), Σy=1e-3)
    # return AbstractGPSurrogate(train_x, train_y, gp=GP(Matern52Kernel(metric=ManifoldDistance(M)) ∘ FunctionTransform(map_pos_orth)), Σy=1e-6)
end

# Some functions to make Zygote happy.
function ManifoldsBase.exp(M::AbstractSphere, p, X)
    θ = Manifolds.norm(M, p, X)
    return cos(θ) .* p .+ Manifolds.usinc(θ) .* X
end

function Manifolds.get_vector(M::AbstractSphere{ℝ}, p, X, ::DefaultOrthonormalBasis)
    n = manifold_dimension(M)
    p1 = p[1]
    cosθ = abs(p1)
    λ = Manifolds.nzsign(p1, cosθ)
    pend = view(p, 2:(n + 1))
    pX = dot(pend, X)
    factor = pX / (1 + cosθ)
    Y = vcat(-λ * pX, X .- pend .* factor)
    return Y
end

struct ZygotePreferredDiffBackend{B<:ManifoldDiff.AbstractDiffBackend} <: ManifoldDiff.AbstractDiffBackend
    backup_backend::B
end
function ManifoldDiff._gradient(f, p, b::ZygotePreferredDiffBackend)
    res = only(Zygote.gradient(f, p))
    any(isnan.(res)) && return ManifoldDiff._gradient(f, p, b.backup_backend)
    return res   
end

function ManifoldDiff._gradient!(f, X, p, b::ZygotePreferredDiffBackend)
    return copyto!(X, ManifoldDiff._gradient(f, p, b))
end

function manifold_surrogate_opt_hybrid(M::AbstractManifold, obj::Function, surr; num_starts = 5*Threads.nthreads())
    # Estimate the standard deviation of the observations and expected values
    # eps = std(surr.y .- surr.(surr.x))

    aq_obj(p) = lower_conf_band(p, surr)
    local man_aq_obj(::AbstractManifold, p) = aq_obj(p)
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ZygotePreferredDiffBackend(ManifoldDiff.FiniteDifferencesBackend())))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))
    local D_man_aq_obj(m::AbstractManifold, p) = ManifoldDiff.gradient(m, aq_obj, p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.ZygoteDiffBackend()))

    # Find the point on the surrogate (and the coordinate) that corresponds to the best known objective value
    min_idx = argmin(surr.y)
    x_min = surr.x[min_idx]

    # Adapt exploration scale such that expected proposal is the distance to the closest points to the current minimiser
    exp_scale = quantile([Manifolds.distance(M, x_min, p) for p in surr.x], 0.1)

    # Sample lots of points from the design space. These serve as starting points when optimising the EI.
    new_samples = [allocate_result(M, rand) for _ in 1:num_starts]
    rand!.(Ref(M), new_samples)

    append!(new_samples, [Manifolds.exp(M, x_min, t) for t in rand_tangent(M, x_min, num_starts; scale=exp_scale)])
    append!(new_samples, [abs.(Manifolds.exp(M, p, only(rand_tangent(M, p; scale=exp_scale)))) for p in StatsBase.sample(surr.x, aweights(softmax(-surr.y)), num_starts)])

    start_points = partialsort(new_samples, 1:num_starts; by=aq_obj)
    Threads.@threads for i = 1:length(start_points)
        start_points[i] = abs.(quasi_Newton(M, man_aq_obj, D_man_aq_obj, start_points[i]))
    end

    new_points = filter(all ∘ Base.Fix1(broadcast, !isnan), start_points)
    @show length(new_points), length(start_points)    
    add_surr_points!(surr, new_points, obj.(new_points))
    # add_surr_points!(surr, new_points, obj.(new_points), eps^2)
    
    simplex_points = partialsort(surr.x, 1:(manifold_dimension(M)+1); by=aq_obj)
    nm_man_obj(::AbstractManifold, p) = obj(p)
    return abs.(Manopt.NelderMead(M, nm_man_obj, NelderMeadSimplex(collect(simplex_points))))
end

function manifold_surrogate_opt_direct(M::AbstractManifold, obj::Function, surr; maxiters = 100, num_new_samples = Threads.nthreads())
    # Minimum distance a new proposal must be from current objective minimum
    dtol = π/1e3
    # Exploration-exploitation trade-off parameter
    eps = std(surr.y .- surr.(surr.x))
    # Count how many failed searches in a row
    C_fails = 0

    aq_fn(p) = surr(p)
    local aq_obj(::Manifolds.Sphere, p) = aq_fn(p)
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ZygotePreferredDiffBackend(ManifoldDiff.FiniteDifferencesBackend())))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))
    local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.ZygoteDiffBackend()))
    # local ZD_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.ZygoteDiffBackend()))
    # local ZPD_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ZygotePreferredDiffBackend(ManifoldDiff.FiniteDifferencesBackend())))
    # local H_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.hessian(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.ZygoteDiffBackend()))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.RiemannianProjectionBackend(ManifoldDiff.FiniteDifferencesBackend()))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.RiemannianProjectionBackend(ManifoldDiff.ZygoteDiffBackend()))

    initial_points = Vector{typeof(first(surr.x))}(undef, length(surr.x))
    # for i = 1:length(surr.x)
    Threads.@threads for i = 1:length(surr.x)
        initial_points[i] = abs.(quasi_Newton(M, aq_obj, D_aq_obj, surr.x[i]))
    end
    # for i = findall(any, map.(isnan, initial_points))
    #     @show surr.x[i]
    #     @show initial_points[i]
    # end

    initial_optima = initial_points[all.(map.(!isnan, initial_points))]
    add_surr_points!(surr, initial_optima, obj.(initial_optima), eps^2)

    for _ in 1:maxiters
        if C_fails >= 3
            println("Cannot find any better points. Exiting.")
            return
        end

        # Evaluate surrogate at all input points
        f_hat = surr.(surr.x)
        # Find the maximum, minimum of the predicted function
        f_min, f_max = extrema(f_hat)
        # Estimate the standard deviation of the observations and expected values
        eps = std(surr.y .- f_hat)

        # Find the point on the surrogate (and the coordinate) that corresponds to the best known objective value
        min_idx = argmin(surr.y)
        x_min = surr.x[min_idx]
        @show surr.y[min_idx]

        # Use the predicted value of the surrogate at this "best" point
        f_min = f_hat[min_idx]
        # Also store which point (that we know about) corresponds to the minimum of the surrogate
        s_min = surr.x[argmin(f_hat)]

        # Adapt exploration scale such that expected proposal is the distance to the closest points to the current minimiser
        exp_scale = quantile([Manifolds.distance(M, x_min, p) for p in surr.x], 0.2)
    
        # Sample lots of points from the design space. These serve as starting points when optimising the EI.
        new_samples = [allocate_result(M, rand) for _ in 1:num_new_samples]
        rand!.(Ref(M), new_samples)

        append!(new_samples, [Manifolds.exp(M, x_min, t) for t in rand_tangent(M, x_min, num_new_samples; scale=exp_scale*(C_fails+1))])
        append!(new_samples, [Manifolds.exp(M, s_min, t) for t in rand_tangent(M, s_min, num_new_samples; scale=exp_scale*(C_fails+1))])
        append!(new_samples, [abs.(Manifolds.exp(M, p, only(rand_tangent(M, p; scale=exp_scale*(C_fails+1))))) for p in StatsBase.sample(surr.x, aweights(softmax(-f_hat)), num_new_samples)])

        # start_points = partialsort(new_samples, 1:1; by=surr)
        start_points = partialsort(new_samples, 1:num_new_samples; by=surr)
        exp_decrease = zeros(Float64, length(start_points))
        prop_points = Vector{typeof(first(surr.x))}(undef, length(start_points))

        # for i = 1:length(start_points)
        Threads.@threads for i = 1:length(start_points)
            res = abs.(quasi_Newton(M, aq_obj, D_aq_obj, start_points[i]))
            exp_decrease[i], prop_points[i] = aq_obj(M, res), res
        end
    
        @show count(any, map.(isnan, prop_points))

        # Filter out points that are too close to one that's already been sampled.
        potential_points = findall(all, eachcol(pairwise(ManifoldDistance(M), surr.x, prop_points) .> dtol))
        if isempty(potential_points)
            # All points we found are too close
            C_fails += 1
            continue
        end

        # Investigate the best point.
        @show best_candidate = findmin(exp_decrease[potential_points])
        if (f_min - first(best_candidate)) < 1e-6*(f_max - f_min)
            # The best point we found didn't improve the answer enough.
            C_fails += 1
            continue
        end

        # If we're here, then it's worth evaluating the objective
        C_fails = 0
        add_surr_points!(surr, prop_points[potential_points], obj.(prop_points[potential_points]), eps^2)
        # new_x = prop_points[potential_points][last(best_candidate)]
        # add_surr_point!(surr, new_x, obj(new_x), eps^2)
    end
    println("Completed maximum number of iterations.")
    return
end

function manifold_surrogate_opt_ei(M::AbstractManifold, obj::Function, surr; maxiters = 100, num_new_samples = Threads.nthreads())
    # Minimum distance a new proposal must be from current objective minimum
    dtol = π/1e3
    # Exploration-exploitation trade-off parameter
    eps = std(surr.y .- surr.(surr.x))
    # Count how many failed searches in a row
    C_fails = 0
    # Current minimum
    f_min = minimum(surr.y)

    aq_fn(p) = log_prob_decrease(p, surr, f_min)
    local aq_obj(::Manifolds.Sphere, p) = -aq_fn(p)

    # aq_fn(p) = expected_decrease(p, surr, f_min, 0.01)
    # local aq_obj(::Manifolds.Sphere, p) = -log(aq_fn(p))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ZygotePreferredDiffBackend(ManifoldDiff.FiniteDifferencesBackend())))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))
    local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.ZygoteDiffBackend()))
    # local ZD_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.ZygoteDiffBackend()))
    # local ZPD_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ZygotePreferredDiffBackend(ManifoldDiff.FiniteDifferencesBackend())))
    # local H_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.hessian(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.ZygoteDiffBackend()))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.RiemannianProjectionBackend(ManifoldDiff.FiniteDifferencesBackend()))
    # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, Base.Fix1(aq_obj, m), p, ManifoldDiff.RiemannianProjectionBackend(ManifoldDiff.ZygoteDiffBackend()))

    # initial_points = Vector{typeof(first(surr.x))}(undef, length(surr.x))
    # # for i = 1:length(surr.x)
    # Threads.@threads for i = 1:length(surr.x)
    #     initial_points[i] = abs.(quasi_Newton(M, aq_obj, D_aq_obj, surr.x[i]))
    # end
    # for i = findall(any, map.(isnan, initial_points))
    #     @show surr.x[i]
    #     @show initial_points[i]
    # end

    # initial_optima = initial_points[all.(map.(!isnan, initial_points))]
    # add_surr_points!(surr, initial_optima, obj.(initial_optima), eps^2)

    for _ in 1:maxiters
        if C_fails >= 3
            println("Cannot find any better points. Exiting.")
            return
        end

        # Evaluate surrogate at all input points
        f_hat = surr.(surr.x)
        # Find the maximum, minimum of the predicted function
        f_min, f_max = extrema(f_hat)
        # Estimate the standard deviation of the observations and expected values
        eps = std(surr.y .- f_hat)

        # Find the point on the surrogate (and the coordinate) that corresponds to the best known objective value
        min_idx = argmin(surr.y)
        @show x_min = surr.x[min_idx]
        @show surr.y[min_idx]

        # Use the predicted value of the surrogate at this "best" point
        f_min = f_hat[min_idx]
        # Also store which point (that we know about) corresponds to the minimum of the surrogate
        @show s_min = surr.x[argmin(f_hat)]

        # Adapt exploration scale such that expected proposal is the distance to the closest points to the current minimiser
        exp_scale = quantile([Manifolds.distance(M, x_min, p) for p in surr.x], 0.05)
    
        # Sample lots of points from the design space. These serve as starting points when optimising the EI.
        new_samples = [allocate_result(M, rand) for _ in 1:num_new_samples]
        rand!.(Ref(M), new_samples)

        append!(new_samples, [Manifolds.exp(M, x_min, t) for t in rand_tangent(M, x_min, num_new_samples; scale=exp_scale*(C_fails+1))])
        append!(new_samples, [Manifolds.exp(M, s_min, t) for t in rand_tangent(M, s_min, num_new_samples; scale=exp_scale*(C_fails+1))])
        append!(new_samples, [abs.(Manifolds.exp(M, p, only(rand_tangent(M, p; scale=exp_scale*(C_fails+1))))) for p in StatsBase.sample(surr.x, aweights(softmax(-f_hat)), num_new_samples)])

        # start_points = partialsort(new_samples, 1:1; by=Base.Fix1(aq_obj, M))
        start_points = partialsort(new_samples, 1:num_new_samples; by=Base.Fix1(aq_obj, M))
        exp_decrease = zeros(Float64, length(start_points))
        prop_points = Vector{typeof(first(surr.x))}(undef, length(start_points))

        # for i = 1:length(start_points)
        Threads.@threads for i = 1:length(start_points)
            res = abs.(quasi_Newton(M, aq_obj, D_aq_obj, start_points[i]))
            exp_decrease[i], prop_points[i] = aq_obj(M, res), res
            # exp_decrease[i], prop_points[i] = aq_fn(res), res
        end
    
        @show count(any, map.(isnan, prop_points))

        # Filter out points that are too close to one that's already been sampled.
        potential_points = findall(all, eachcol(pairwise(ManifoldDistance(M), surr.x, prop_points) .> dtol))
        if isempty(potential_points)
            # All points we found are too close
            C_fails += 1
            continue
        end

        # Investigate the best point.
        @show best_candidate = findmin(exp_decrease[potential_points])
        if exp(-first(best_candidate)) < 1e-3
        # if first(best_candidate) < 1e-6*(f_max - f_min)
            # The best point we found didn't improve the answer enough.
            C_fails += 1
            continue
        end

        # If we're here, then it's worth evaluating the objective
        C_fails = 0
        add_surr_points!(surr, prop_points[potential_points], obj.(prop_points[potential_points]), eps^2)
        # new_x = prop_points[potential_points][last(best_candidate)]
        # add_surr_point!(surr, new_x, obj(new_x), eps^2)
    end
    println("Completed maximum number of iterations.")
    return
end

# fixed bayes optimisation algorithm.
function surrsphereopt(obj::Function, surr; maxiters = 100, num_new_samples = Threads.nthreads())
    # infer dimension from surrogate x values
    d = length(first(surr.x))
    M = Manifolds.Sphere(Val(d - 1))

    # Minimum distance a new proposal must be from current objective minimum
    dtol = π/1e3
    # Exploration-exploitation trade-off parameter
    eps = 0.01
    # Count how many failed searches in a row
    C_fails = 0

    for _ in 1:maxiters
        if C_fails >= 3
            println("Cannot find any better points. Exiting.")
            return
        end

        # Evaluate surrogate at all input points
        f_hat = surr.(surr.x)
        # Find the maximum, minimum of the predicted function
        f_min, f_max = extrema(f_hat)
        # Estimate the standard deviation of the observations and expected values
        eps = std(surr.y .- f_hat)

        # Find the point on the surrogate (and the coordinate) that corresponds to the best known objective value
        min_idx = argmin(surr.y)
        x_min = surr.x[min_idx]
        @show surr.y[min_idx]

        # Use the predicted value of the surrogate at this "best" point
        f_min = f_hat[min_idx]
        # Also store which point (that we know about) corresponds to the minimum of the surrogate
        s_min = surr.x[argmin(f_hat)]

        aq_obj(p) = -log(expected_decrease(p, surr, f_min, eps))
        aq_obj(::Manifolds.Sphere, p) = aq_obj(p)
        D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, aq_obj, p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))
        # local aq_obj(::Manifolds.Sphere, p) = -log(expected_decrease(p, surr, f_min, eps))
        # local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, y->aq_obj(m, p), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))

        # Adapt exploration scale such that expected proposal is the distance to the closest points to the current minimiser
        exp_scale = quantile([Manifolds.distance(M, x_min, p) for p in surr.x], 0.2)

        # Sample lots of points from the design space. These serve as starting points when optimising the EI.
        new_samples = Vector{typeof(first(surr.x))}(undef, 0)
        append!(new_samples, [Manifolds.exp(M, x_min, t) for t in rand_tangent(M, x_min, num_new_samples; scale=exp_scale)])
        append!(new_samples, [Manifolds.exp(M, s_min, t) for t in rand_tangent(M, s_min, num_new_samples; scale=exp_scale)])
        append!(new_samples, [Manifolds.exp(M, p, only(rand_tangent(M, p; scale=exp_scale))) for p in StatsBase.sample(surr.x, aweights(softmax(-aq_obj.(Ref(M), surr.x))), num_new_samples)])
        append!(new_samples, rand(M, num_new_samples))
        # start_points = new_samples[partialsortperm(new_samples, 1:1, by=Base.Fix1(aq_obj, M))]
        start_points = new_samples[partialsortperm(new_samples, 1:num_new_samples, by=Base.Fix1(aq_obj, M))]
        # start_points = new_samples
        exp_decrease = zeros(Float64, length(start_points))
        prop_points = Vector{typeof(first(surr.x))}(undef, length(start_points))

        # for i = 1:length(start_points)
        Threads.@threads for i = 1:length(start_points)
            res = abs.(quasi_Newton(M, aq_obj, D_aq_obj, start_points[i]))
            # res = abs.(start_points[i])
            exp_decrease[i], prop_points[i] = exp(-aq_obj(M, res)), res
        end

        # Filter out points that are too close to one that's already been sampled.
        potential_points = findall(all, eachcol(pairwise(euclidean, surr.x, prop_points) .> dtol))
        @show exp_decrease[potential_points]
        if isempty(potential_points)
            # All points we found are too close
            C_fails += 1
            continue
        end

        # Investigate the best point.
        @show best_candidate = findmax(exp_decrease[potential_points])
        if first(best_candidate) < 1e-6*(f_max - f_min)
            # The best point we found didn't improve the answer enough.
            C_fails += 1
            continue
        end

        # If we're here, then it's worth evaluating the objective
        C_fails = 0
        # add_surr_points!(surr, prop_points[potential_points], obj.(prop_points[potential_points]), eps^2)
        new_x = prop_points[potential_points][last(best_candidate)]
        add_surr_point!(surr, new_x, obj(new_x), eps^2)
    end
    println("Completed maximum number of iterations.")
    return
end

# fixed bayes optimisation algorithm.
function bayessphereopt(obj::Function, surr; maxiters = 100, num_new_samples = Threads.nthreads())
    # infer dimension from surrogate x values
    d = length(first(surr.x))
    M = Manifolds.Sphere(Val(d - 1))

    # Minimum distance a new proposal must be from current objective minimum
    dtol = π/1e3
    # Exploration-exploitation trade-off parameter
    eps = 0.01
    # Count how many failed searches in a row
    C_fails = 0

    for _ in 1:maxiters
        if C_fails >= 3
            println("Cannot find any better points. Exiting.")
            return
        end

        # Evaluate surrogate at all input points
        f_hat = surr.(surr.x)
        # Find the maximum, minimum of the predicted function
        f_min, f_max = extrema(f_hat)
        # Estimate the standard deviation of the observations and expected values
        eps = std(surr.y .- f_hat)

        # Find the point on the surrogate (and the coordinate) that corresponds to the best known objective value
        min_idx = argmin(surr.y)
        x_min = surr.x[min_idx]
        @show surr.y[min_idx]

        # Use the predicted value of the surrogate at this "best" point
        f_min = f_hat[min_idx]
        # Also store which point (that we know about) corresponds to the minimum of the surrogate
        s_min = surr.x[argmin(f_hat)]

        local aq_obj(::Manifolds.Sphere, p) = -log(expected_decrease(p, surr, f_min, eps))
        local D_aq_obj(m::Manifolds.Sphere, p) = ManifoldDiff.gradient(m, y->aq_obj(m, p), p, ManifoldDiff.TangentDiffBackend(ManifoldDiff.FiniteDifferencesBackend()))

        # Adapt exploration scale such that expected proposal is the distance to the closest points to the current minimiser
        exp_scale = quantile([Manifolds.distance(M, x_min, p) for p in surr.x], 0.2)

        # Sample lots of points from the design space. These serve as starting points when optimising the EI.
        new_samples = Vector{typeof(first(surr.x))}(undef, 0)
        append!(new_samples, [Manifolds.exp(M, x_min, t) for t in rand_tangent(M, x_min, num_new_samples; scale=exp_scale)])
        append!(new_samples, [Manifolds.exp(M, s_min, t) for t in rand_tangent(M, s_min, num_new_samples; scale=exp_scale)])
        append!(new_samples, [Manifolds.exp(M, p, only(rand_tangent(M, p; scale=exp_scale))) for p in StatsBase.sample(surr.x, aweights(exp.(-aq_obj.(Ref(M), surr.x))), num_new_samples)])
        append!(new_samples, rand(M, num_new_samples))
        start_points = new_samples[partialsortperm(new_samples, 1:num_new_samples, by=Base.Fix1(aq_obj, M))]
        exp_decrease = zeros(Float64, length(start_points))
        prop_points = Vector{typeof(first(surr.x))}(undef, length(start_points))

        Threads.@threads for i = 1:length(start_points)
            res = abs.(quasi_Newton(M, aq_obj, D_aq_obj, start_points[i]))
            # res = abs.(start_points[i])
            exp_decrease[i], prop_points[i] = exp(-aq_obj(M, res)), res
        end

        # Filter out points that are too close to one that's already been sampled.
        potential_points = findall(all, eachcol(pairwise(euclidean, surr.x, prop_points) .> dtol))
        if isempty(potential_points)
            # All points we found are too close
            C_fails += 1
            continue
        end

        # Investigate the best point.
        @show best_candidate = findmax(exp_decrease[potential_points])
        if first(best_candidate) < 1e-6*(f_max - f_min)
            # The best point we found didn't improve the answer enough.
            C_fails += 1
            continue
        end

        # If we're here, then it's worth evaluating the objective
        C_fails = 0
        new_x = prop_points[potential_points][last(best_candidate)]
        add_surr_point!(surr, new_x, obj(new_x), eps^2)
    end
    println("Completed maximum number of iterations.")
    return
end

# Quasi-random sampler implementation for use with Surrogates.jl optimisers.
struct StratifiedSphericalSampler{D} <: QuasiMonteCarlo.SamplingAlgorithm end
QuasiMonteCarlo.sample(n_pts::Integer, ::Union{Number, Tuple, AbstractVector}, ::Union{Number, Tuple, AbstractVector}, ::StratifiedSphericalSampler{D}) where {D} = Tuple.(eachcol(stratified_sphere_sample(D, n_pts)))

function sphereopt(obj::Function, ::EI, krig; maxiters = 100, num_new_samples = 96)
    # Dimension of the space
    d = length(first(krig.x)) + 1

    # Minimum distance a new proposal must be from current objective minimum"
    dtol = 1e-3 * pi
    
    # Exploration-exploitation trade-off parameter
    eps = 0.01

    # Count how many failed searches in a row
    C_fails = 0

    for _ in 1:maxiters
        if C_fails >= 3
            println("Cannot find any better points. Exiting.")
            return
        end

        # Find the maximum, minimum so far
        f_min, f_max = extrema(krig.y)

        # Sample lots of points from the design space. These serve as starting points when optimising the EI.
        new_sample = stratified_sphere_sample(d, num_new_samples)

        # Maximize the expected improvement for each starting point and store the results (EI, x').
        exp_decrease = zeros(Float64, num_new_samples)
        prop_points = Vector{typeof(first(krig.x))}(undef, num_new_samples)
        # for i = 1:num_new_samples
        #     prop_points[i] = new_sample[i]
        #     exp_decrease[i] = -expected_improvement(prop_points[i], krig, f_min, eps)
        # end
        Threads.@threads for i = 1:num_new_samples
            # Don't bother optimising if there's no potential benefit starting here.
            if expected_improvement(new_sample[i], krig, f_min, eps) ≈ 0
                exp_decrease[i], prop_points[i] = 0, new_sample[i]
            else
                res = Optim.optimize(x -> -expected_improvement(Ω⁻¹(x), krig, f_min, eps), Ω(new_sample[i]), LBFGS(linesearch = LineSearches.BackTracking(), manifold=Optim.Sphere()));
                exp_decrease[i], prop_points[i] = -Optim.minimum(res), Tuple(Ω⁻¹(Optim.minimizer(res)))
            end
        end

        # Filter out points that are too close to one that's already been sampled.
        potential_points = findall(all, eachcol([Manifolds.distance(Manifolds.Sphere(Val(d - 1)), Ω(cx), Ω(px)) .> dtol for cx in krig.x, px in prop_points]))
        if isempty(potential_points)
            # All points we found are too close
            C_fails += 1
            continue
        end

        # Investigate the best point.
        best_candidate = findmax(exp_decrease[potential_points])
        if first(best_candidate) < 1e-6*(f_max - f_min)
            # The best point we found didn't improve the answer enough.
            C_fails += 1
            continue
        end

        # If we're here, then it's worth evaluating the objective
        C_fails = 0
        new_x = prop_points[potential_points][last(best_candidate)]
        add_point!(krig, Tuple(new_x), obj(new_x))
    end
    println("Completed maximum number of iterations.")
    return
end

# manifold differential evolution:
function manifold_diffevo(M::AbstractManifold, obj::Function, niters::Int=100, n_pop::Int=100; diff_wt=0.8, cr_rate=0.9)
    pop = [allocate_result(M, identity) for _ in 1:n_pop]
    rand!.(Ref(M), pop)
    # pop = map.(abs, pop)
    f_pop = Vector{Float64}(undef, n_pop)
    f_pop .= obj.(pop)
    f_range = diff([extrema(f_pop)...])
    B = DefaultBasis()

    #! todo: stopping criterion.
    for _ in 1:niters
        next_gen = similar(pop)
        f_next = similar(f_pop)
        let pop=pop, f_pop=f_pop
            # for I = 1:n_pop
            Threads.@threads for I = 1:n_pop
                # Random parent points
                pt_a, pt_b, pt_c = getindex(pop, rand(filter(!=(I), 1:n_pop), 3))
        
                # Point being mutated.
                x = pop[I]
        
                # mutation vector, initialised to 0s, e.g. stay at x.
                m = zeros(manifold_dimension(M))
                # Add movement to pt_a
                m .+= get_coordinates(M, x, inverse_retract(M, x, pt_a), B)
                # m .+= get_coordinates(M, x, log(M, x, pt_a), B)
        
                # Compute tangent vector at x that is the transport of the move from pt_c to pt_b.
                tx_vcb = vector_transport_to(M, pt_c, inverse_retract(M, pt_c, pt_b), x)
                # tx_vcb = parallel_transport_to(M, pt_c, log(M, pt_c, pt_b), x)
                # Add this.
                m .+= diff_wt*get_coordinates(M, x, tx_vcb, B) #? I think there shouldn't be another retract here.
                # m .+= diff_wt*get_coordinates(M, x, inverse_retract(M, x, tx_vcb), B)
                # @show m .+= diff_wt*get_coordinates(M, x, log(M, x, tx_vcb), B)
                #? Can't avoid using basis because of ProductRepr later on.
        
                #? Todo: Implement Radius Limited Version
                gi = rand(1:manifold_dimension(M))
                tv = similar(m)
                #? todo: make this more efficient.
                for i in 1:manifold_dimension(M)
                    if (i == gi) || (rand() < cr_rate)
                        tv[i] = m[i]
                    else
                        tv[i] = 0
                    end
                end
                # xp = abs.(retract(M, x, get_vector(M, x, tv, B)))
                xp = retract(M, x, get_vector(M, x, tv, B))
                # xp = exp(M, x, get_vector(M, x, tv, B))
                fp = obj(xp)
                if (fp <= f_pop[I])
                    next_gen[I] = xp
                    f_next[I] = fp
                else
                    next_gen[I] = x
                    f_next[I] = f_pop[I]
                end
            end
        end
        pop .= next_gen
        f_pop .= f_next

        if (diff([extrema(f_pop)...]) <= 1e-3*f_range)
            break
        end
    end

    return pop[argmin(f_pop)]
end

