struct Percentage
    p::Float64

    function Percentage(p::Real)
        pp = convert(Float64, p/100)
        0.0 < pp ≤ 1.0 || error("Percentage should be in (0, 1]. Please recall \
                                this function converts a desired percentage \
                                by dividing it by 100.")
        return new(pp)
    end
end

Base.rem(p::Real) = Percentage(p)
(k::Percentage)(N::Integer)::Int = round(Int, N*k.p)

# function partial(f::Function; kwargs...)
#     function(args...; override...)
#         f(args...; kwargs..., override...)
#     end
# end

# Does the conversion from the Φ to S (always on unit-sphere)
function Ω(ϕ)
    res::Vector{Float64} = ones(length(ϕ)+1)

    res[1:end-1] .*= cos.(ϕ)
    res[2:end]   .*= cumprod(sin.(ϕ))

    return res
end

# function Ω⁻¹(x)
#     res::Vector{Float64} = reverse(sqrt.(cumsum(reverse(x.^2))))

#     @. res[2:end] = acos(x[1:end-1]/res[1:end-1])
#     res[isnan.(res)] .= 0.0
#     res[end] = mod(2pi + sign(x[end])*res[end], 2pi)

#     return res[2:end]
# end

# Inverts the mapping form Φ to S (drops radius)
function Ω⁻¹(x)
    res::Vector{Float64} = reverse(sqrt.(cumsum(reverse(x.^2))))
    zs = iszero.(res)

    @. res[2:end] = acos(x[1:end-1]/res[1:end-1])
    if any(zs)
        res[zs] .= 0.0
        nz = count(zs)
        res[end - nz + 1] += (1 - sign(x[end-nz]))*pi/2
    end
    if x[end] < 0
        res[end] += 2(π - res[end])
    end

    return res[2:end]
end

# Other includes
include("./intractable_density_term.jl")
include("./model.jl")
include("./kernel_transforms.jl")
include("./approximator.jl")
include("./distance_estimators.jl")
include("./adaptive_kernels.jl")
include("./optimisation.jl")
