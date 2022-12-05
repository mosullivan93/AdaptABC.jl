module AdaptABC

using Random
using Distances
using Distributions
using LogExpFunctions
using Manifolds
using Manopt

export
    # The implicit model and associate algebraic tools
    ImplicitDistribution,
    IntractableLikelihood,
    IntractableTerm,
    IntractableExpression,

    prior,
    paramnames,
    summarise,
    simulate,
   
    ImplicitBayesianModel,
    ImplicitPosterior,
    PosteriorApproximator,
    Kernel,
    KernelRecipe,
    ProductKernel,

    # Samplers
    rejection_sampler,
    mcmc_sampler,
    smc_sampler,
    adaptive_smc_sampler_MAD,
    adaptive_smc_sampler_opt_oaat,
    adaptive_smc_sampler_opt_dfo,

    # Transformations
    ScalingTransform


# Includes
include("./interface/common.jl")
include("./samplers/common.jl")

end
