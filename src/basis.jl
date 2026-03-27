# the original Splines2 implementation used zero-indexed OffsetArrays
# to ease the port from C
struct SplineBasis{T<:Real} <: AbstractSplineBasis{T}
    order::Int # order of the spline
    nknots::Int # number of knots
    ncoef::Int # number of coefficients
    knots::OffsetArray{T,1} # knot vector
end

function SplineBasis(knots::AbstractVector{T}, order::Int=4) where {T<:Real}
    return SplineBasis(order,
                       length(knots),
                       length(knots) - order,
                       zeroIndexedArray(knots))
end

struct BSplineBasis{T<:Real} <: AbstractSplineBasis{T}
    spline_basis::SplineBasis{T}
    boundary_knots::Tuple{T,T}
    # TODO: refactor so that this can just be an empty vector
    # instead of needing the sentinel nothing 
    # maybe also a static array?
    interior_knots::Union{Array{T,1},Nothing}
    intercept::Bool
    df::Int
end

function BSplineBasis(boundary_knots::Tuple{T,T},
                      interior_knots::Union{Array{T,1},Nothing}=nothing,
                      order::Int=4,
                      intercept::Bool=false) where {T<:Real}
    l_interior_knots = isnothing(interior_knots) ? 0 : length(interior_knots)
    df = Int(intercept) + order - 1 + l_interior_knots
    nknots = l_interior_knots + 2 * order

    # knots are initialized as the boundary knots that they are closest to
    # we start with the 'right' boundary to make the indexing logic easier
    knots = fill(last(boundary_knots), nknots)
    fill!(view(knots, 1:order), first(boundary_knots))

    for i in 1:l_interior_knots
        knots[i + order] = interior_knots[i]
    end

    return BSplineBasis(SplineBasis(knots, order),
                        boundary_knots, interior_knots,
                        intercept, df)
end

function _find_interval(bs::SplineBasis{T}, x::T) where {T<:Real}
    k = bs.order - 1
    n = bs.nknots - k - 1
    l = k
    while x < bs.knots[l] && l != k
        l = l - 1
    end
    l = l + 1
    while x >= bs.knots[l] && l != n
        l = l + 1
    end
    return l - 1
end

function basis(bs::SplineBasis{T}, x::T, derivs::Int=0) where {T<:Real}
    t = bs.knots
    k = bs.order - 1
    m = derivs
    hh = bs.order
    ell = _find_interval(bs, x)
    result = zeroIndexedArray(zeros(T, 2 * k + 2))
    one = T(1)
    zero = T(0)
    result[0] = one
    for j in 1:(k - m)
        for n in 0:(j - 1)
            result[hh + n] = result[n]
        end
        result[0] = zero
        for n in 1:j
            ind = ell + n
            xb = t[ind]
            xa = t[ind - j]
            if (xb == xa)
                result[n] = zero
                continue
            end
            w = result[hh + n - 1] / (xb - xa)
            result[n - 1] = result[n - 1] + w * (xb - x)
            result[n] = w * (x - xa)
        end
    end
    for j in (k - m + 1):k
        for n in 0:(j - 1)
            result[hh + n] = result[n]
        end
        result[0] = zero
        for n in 1:j
            ind = ell + n
            xb = t[ind]
            xa = t[ind - j]
            if xb == xa
                result[m] = zero
                continue
            end
            w = j * result[hh + n - 1] / (xb - xa)
            result[n - 1] = result[n - 1] - w
            result[n] = w
        end
    end
    offset = ell - k
    v = zeros(T, bs.ncoef)
    v[(1 + offset):(k + 1 + offset)] = result[0:k]
    return v
end

function basis(bs::BSplineBasis{T}, x::T, derivs::Int=0) where {T<:Real}
    if bs.boundary_knots[1] <= x <= bs.boundary_knots[2]
        vec = basis(bs.spline_basis, x, derivs)
    else # outside of the boundary knots
        if x < bs.boundary_knots[1]
            k_pivot = T(0.75) * bs.boundary_knots[1] +
                      T(0.25) * bs.spline_basis.knots[bs.spline_basis.order + 1 - 1] # 0-based
        else
            k_pivot = T(0.75) * bs.boundary_knots[2] +
                      T(0.25) *
                      bs.spline_basis.knots[length(bs.spline_basis.knots) - bs.spline_basis.order - 1 - 1] # 0-based
        end
        delta = x - k_pivot
        if derivs == 0
            vec = basis(bs.spline_basis, k_pivot, 0) +
                  basis(bs.spline_basis, k_pivot, 1) * delta +
                  basis(bs.spline_basis, k_pivot, 2) * delta * delta / T(2.0) +
                  basis(bs.spline_basis, k_pivot, 3) * delta * delta * delta / T(6.0)
        elseif derivs == 1
            vec = splines.basis(bs.spline_basis, k_pivot, 1) +
                  splines.basis(bs.spline_basis, k_pivot, 2) * delta +
                  splines.basis(bs.spline_basis, k_pivot, 3) * delta * delta / T(2.0)
        elseif derivs == 2
            vec = splines.basis(bs.spline_basis, k_pivot, 2) +
                  splines.basis(bs.spline_basis, k_pivot, 3) * delta
        elseif derivs == 3
            vec = splines.basis(bs.spline_basis, k_pivot, 3)
        else
            vec = k_pivot .* T(0)
        end
    end
    if !bs.intercept
        vec = vec[2:length(vec)]
    end
    return vec
end

function basis(bs::AbstractSplineBasis{T}, x::AbstractVector{T},
               derivs::Int=0) where {T<:Real}
    f(xi) = basis(bs, xi, derivs)
    return stack(f, x; dims=1)
end

"""
    spline_args(x::AbstractVector{T};
                boundary_knots::Union{Tuple{T,T},Nothing}=nothing,
                interior_knots::Union{Vector{T},Nothing}=nothing,
                order::Int=4,
                intercept::Bool=false,
                df::Int=3 + Int(intercept),
                knots::Union{Vector{T},Nothing}=nothing,
                knots_offset::Int=0) where {T<:Real}

Utility function for processing the spline arguments.

Returns the computed boundary and interior knots. If these 
are already both computed (i.e. not nothing), then returns
the passed values.
"""
function spline_args(x::AbstractVector{T},
                     boundary_knots::Tuple{T,T},
                     interior_knots::Vector{T};
                     kwargs...) where {T<:Real}
    return (; boundary_knots, interior_knots)
end

function spline_args(x::AbstractVector{T},
                     boundary_knots::Union{Tuple{T,T},Nothing},
                     interior_knots::Union{Vector{T},Nothing};
                     order::Int=4,
                     intercept::Bool=false,
                     df::Int=3 + Int(intercept),
                     knots::Union{Vector{T},Nothing}=nothing,
                     knots_offset::Int=0) where {T<:Real}
    if !isnothing(knots)
        length(knots) == 1 &&
            error("At least two knots are required if knots are specified.")
        # if knots are passed, then use the first and last knots as
        # the boundary knots and any remaining knots as interior knots
        # NOTE: this assumes that the knots are already sorted so that the boundary knots
        # are first and last
        boundary_knots = (first(knots), last(knots))
        interior_knots = length(knots) == 2 ? nothing : knots[2:(length(knots) - 1)]
    else
        boundary_knots = @something(boundary_knots, extrema(x))
        iKnots = df - order + knots_offset + 1 - Int(intercept)
        if iKnots > 0
            # we exclude the 0th and 100th percentiles
            p = view(range(T(0); length=iKnots + 2, stop=T(1)), 2:(iKnots + 1))
            interior = boundary_knots[1] .<= x .<= boundary_knots[2]
            interior_knots = quantile(view(x, interior), p)
        end
    end
    return (; boundary_knots, interior_knots)
end
