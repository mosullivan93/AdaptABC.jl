"A placeholder datatype that represents the unevaluated likelihood"
struct IntractableLikelihood{M}
    obs
    params

    "Input observations and the model instance"
    IntractableLikelihood(o, m) = IntractableTerm{:Identity}(new{typeof(m)}(o, params(m)))
end
Base.:(==)(ilL::IntractableLikelihood{M1}, ilR::IntractableLikelihood{M2}) where {M1, M2} = isequal(M1, M2) && isequal(ilL.params, ilR.params) && isequal(ilL.obs, ilR.obs)
Base.hash(il::IntractableLikelihood{M}, h::UInt) where {M} = hash(M, hash(il.params, hash(il.obs, h)))

"Provides an ordering such that terms can be sorted"
function intractable_sort_order(ilL::IntractableLikelihood{M1}, ilR::IntractableLikelihood{M2}) where {M1, M2}
    # Sort on model
    isless(string(M1), string(M2)) && return true
    if isequal(M1, M2)
        # Sort on params
        isless(ilL.params, ilR.params) && return true
        if isequal(ilL.params, ilR.params)
            # Sort on observations
            return isless(ilL.obs, ilR.obs)
        end
    end
    return false
end

"A placeholder datatype that represents the application of T to a term, then its exponentiation and multiplication"
struct IntractableTerm{T}
    term
    _power::Float64
    _mult::Float64

    function IntractableTerm{T}(t; _p=float(1), _m=float(1)) where {T}
        isinf(_p) && error("undefined") # Might work if t <= 1, undefined without this knowledge
        isinf(_m) && error("undefined") # Might work if t == 0, undefined without this knowledge

        iszero(_m) && return float(0)
        iszero(_p) && return _m

        return new{T}(t, _p, _m)
    end
end
Base.:(==)(itL::IntractableTerm{T1}, itR::IntractableTerm{T2}) where {T1, T2} = isequal(T1, T2) && isequal(itL.term, itR.term) && isequal(itL._power, itR._power) && isequal(itL._mult, itR._mult)
Base.hash(it::IntractableTerm{T}, h::UInt) where {T} = hash(T, hash(it.term, hash(it._power, hash(it._mult, h))))

"Provides an ordering such that terms can be sorted"
function intractable_sort_order(itL::IntractableTerm{T1}, itR::IntractableTerm{T2}) where {T1, T2}
    # Sort on transformation
    isless(string(T1), string(T2)) && return true
    if isequal(T1, T2)
        # Sort on term
        intractable_sort_order(itL.term, itR.term) && return true
        if isequal(itR.term, itR.term)
            # Sort on power
            isless(itL._power, itR._power) && return true
            if isequal(itR._power, itR._power)
                # Sort on factor
                return isless(itL._mult, itR._mult)
            end
        end
    end
    return false
end

Base.:(-)(it::IntractableTerm{T}) where {T} = IntractableTerm{T}(it.term, _p = it._power, _m = -it._mult)

alike(itL::IntractableTerm{T1}, itR::IntractableTerm{T2}) where {T1, T2} = isequal(T1, T2) && isequal(itL.term, itR.term)

function Base.log(it::IntractableTerm{:Identity})
    local lT = IntractableTerm{:Log}(it.term, _m=it._power)
    local lC = log(it._mult)
    return iszero(lC) ? lT : IntractableExpression([lT], lC)
end

function Base.exp(it::IntractableTerm{:Log})
    isone(it._power) || error("Can't exponentiate exponentiated logarithm: $(it._power)")
    return IntractableTerm{:Identity}(it.term, _p=it._mult)
end

Base.:(^)(it::IntractableTerm{T}, pow::Real) where {T} = IntractableTerm{T}(it.term, _p=it._power*pow, _m=it._mult^pow)
Base.:(*)(it::IntractableTerm, mult) = mult*it
Base.:(*)(mult, it::IntractableTerm{T}) where {T} = IntractableTerm{T}(it.term, _p=it._power, _m=mult*it._mult)
Base.:(/)(it::IntractableTerm, dvsr) = (1/dvsr) * it
Base.:(/)(mult, it::IntractableTerm{T}) where {T} = IntractableTerm{T}(it.term, _p=-it._power, _m=mult/it._mult)

function Base.:(*)(itL::IntractableTerm{T1}, itR::IntractableTerm{T2}) where {T1, T2}
    alike(itL, itR) || error("Multiplying distinct terms/transformations is not supported.")

    return IntractableTerm{T1}(itL.term, _p = itL._power + itR._power, _m = itL._mult * itR._mult)
end
function Base.:(/)(itL::IntractableTerm{T1}, itR::IntractableTerm{T2}) where {T1, T2}
    alike(itL, itR) || error("Dividing distinct terms/transformations is not supported.")

    return IntractableTerm{T1}(itL.term, _p = itL._power - itR._power, _m = itL._mult / itR._mult)
end
Base.:(+)(it::IntractableTerm, term) = IntractableExpression([it], term)
Base.:(+)(term, it::IntractableTerm) = it + term
function Base.:(+)(itL::IntractableTerm{T1}, itR::IntractableTerm{T2}) where {T1, T2}
    alike(itL, itR) && isequal(itL._power, itR._power) && return IntractableTerm{T1}(itL.term, _p = itL._power, _m = itL._mult + itR._mult)

    return IntractableExpression([itL, itR], 0)
end
Base.:(-)(itL::IntractableTerm, term) = itL + (-term)
Base.:(-)(itL::IntractableTerm, itR::IntractableTerm) = itL + (-itR)
Base.:(-)(term, itL::IntractableTerm) = (-itL) + term

"A placeholder datatype that represents a summation of one or more intractable terms and a constant"
struct IntractableExpression
    terms
    _add::Float64

    function IntractableExpression(ts, a)
        isone(length(ts)) && iszero(a) && return first(ts)
        isempty(ts) && return a

        # On second thought, I'm not sure sorting is necessary.
        return new(sort(ts, lt=intractable_sort_order), a)
    end
end
Base.:(==)(ieL::IntractableExpression, ieR::IntractableExpression) = isequal(ieL._add, ieR._add) && isequal(ieL.terms, ieR.terms)
Base.hash(ie::IntractableExpression, h::UInt) = hash(ie.terms, hash(ie._add, h))

Base.:(-)(ie::IntractableExpression) = IntractableExpression(map((-), ie.terms), -ie._add)

Base.:(+)(ie::IntractableExpression, term) = IntractableExpression(ie.terms, ie._add + term)
Base.:(-)(ie::IntractableExpression, term) = IntractableExpression(ie.terms, ie._add - term)
Base.:(+)(it::IntractableTerm, ie::IntractableExpression) = ie + it
function Base.:(+)(ie::IntractableExpression, it::IntractableTerm{T}) where {T}
    # Should only be one at most...
    local ind = findfirst(Base.Fix1(alike, it), ie.terms)
    isnothing(ind) && return IntractableExpression(vcat(ie.terms, it), ie._add)
    # Add the modified term back to an expression containing the rest
    return IntractableExpression([t for (i, t) in enumerate(ie.terms) if i != ind], ie._add) + (ie.terms[ind] + it)
end
Base.:(-)(ie::IntractableExpression, it::IntractableTerm) = ie + (-it)
Base.:(+)(ieL::IntractableExpression, ieR::IntractableExpression) = foldl((+), ieR.terms, init=ieL + ieR._add)
Base.:(-)(ieL::IntractableExpression, ieR::IntractableExpression) = ieL + (-ieR)
Base.:(*)(term, ie::IntractableExpression) = IntractableExpression(map(Base.Fix1((*), term), ie.terms), term*ie._add)
