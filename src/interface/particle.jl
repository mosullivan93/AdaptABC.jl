#? Could the types here be tied in somehow with the model?

# making these tuples complicates things too much...
struct Particle
    θ::Vector{Float64}
    X::Vector{Float64}
    ρ::Float64
    #! update the iterators if adding weight
    # weight?
end

param(p::Particle) = p.θ
summ(p::Particle) = p.X
dist(p::Particle) = p.ρ
# weight(p::Particle) = p.w

param(ps::AbstractVector{<:Particle}) = reduce(hcat, param.(ps))
summ(ps::AbstractVector{<:Particle}) = reduce(hcat, summ.(ps))
dist(ps::AbstractVector{<:Particle}) = dist.(ps)
# weight(ps::AbstractVector{<:Particle}) = weight.(ps)

#! define copy? or expect deepcopy to work?
#? already had to fixup mcmc to ensure I get a copy...
#? be very careful when mutating a particle vector.

# indexed_iterate is used for tuple destructuring
@inline Base.indexed_iterate(p::Particle, ::Int, state::Val{:θ} = Val{:θ}()) = (p.θ, Val{:X}())
@inline Base.indexed_iterate(p::Particle, ::Int, state::Val{:X}) = (p.X, Val{:ρ}())
@inline Base.indexed_iterate(p::Particle, ::Int, state::Val{:ρ}) = (p.ρ, nothing)

@inline Base.indexed_iterate(ps::AbstractVector{Particle}, ::Int, state::Val{:θ} = Val{:θ}()) = (param(ps), Val{:X}())
@inline Base.indexed_iterate(ps::AbstractVector{Particle}, ::Int, state::Val{:X}) = (summ(ps), Val{:ρ}())
@inline Base.indexed_iterate(ps::AbstractVector{Particle}, ::Int, state::Val{:ρ}) = (dist(ps), nothing)
