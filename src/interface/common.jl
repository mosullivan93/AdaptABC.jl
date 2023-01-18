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

include("./intractable_density_term.jl")
include("./model.jl")
include("./kernel_transforms.jl")
include("./approximator.jl")
include("./adaptive_kernels.jl")
include("./distance_estimation.jl")
