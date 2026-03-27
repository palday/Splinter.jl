# Note: we have used OffsetArray for converting from C code
struct SplineBasis{T<:Real} <: AbstractSplineBasis{T}
    order::Int # order of the spline
    nknots::Int # number of knots
    ncoef::Int # number of coefficients
    knots::OffsetArray{T,1} # knot vector
end

struct BSplineBasis{T<:Real} <: AbstractSplineBasis{T}
    spline_basis::SplineBasis{T}
    boundary_knots::Tuple{T,T}
    interior_knots::Union{Array{T,1},Nothing}
    intercept::Bool
    df::Int
end

function BSplineBasis(boundary_knots::Tuple{T,T},
                      interior_knots::Union{Array{T,1},Nothing}=nothing,
                      order::Int=4,
                      intercept::Bool=false) where {T<:Real}
    l_interior_knots = interior_knots == nothing ? 0 : length(interior_knots)
    df = Int(intercept) + order - 1 + l_interior_knots
    nknots = l_interior_knots + 2 * order
    ncoef = nknots - order
    knots = zeros(T, nknots)
    for i in 1:order
        knots[i] = boundary_knots[1]
        knots[nknots - i + 1] = boundary_knots[2]
    end
    if (l_interior_knots > 0)
        for i in 1:l_interior_knots
            knots[i + order] = interior_knots[i]
        end
    end
    return BSplineBasis(SplineBasis(knots, order), boundary_knots, interior_knots,
                        intercept, df)
end


# constructors
function SplineBasis(knots::Array{T,1}, order::Int=4) where {T<:Real}
    return SplineBasis(order,
                       length(knots),
                       length(knots) - order,
                       zeroIndexedArray(knots))
end

function basis(bs::SplineBasis{T}, x::T, ders::Int=0) where {T<:Real}
    function find_interval(x::T)
        k = bs.order - 1
        n = bs.nknots - k - 1
        l = k
        while (x < bs.knots[l] && l != k)
            l = l - 1
        end
        l = l + 1
        while (x >= bs.knots[l] && l != n)
            l = l + 1
        end
        return l - 1
    end
    t = bs.knots
    k = bs.order - 1
    m = ders
    hh = bs.order
    ell = find_interval(x)
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
            if (xb == xa)
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

function basis(bs::BSplineBasis{T}, x::T, ders::Int=0) where {T<:Real}
    if (x < bs.boundary_knots[1] || x > bs.boundary_knots[2])
        if (x < bs.boundary_knots[1])
            k_pivot = T(0.75) * bs.boundary_knots[1] +
                      T(0.25) * bs.spline_basis.knots[bs.spline_basis.order + 1 - 1] # 0-based
        else
            k_pivot = T(0.75) * bs.boundary_knots[2] +
                      T(0.25) *
                      bs.spline_basis.knots[length(bs.spline_basis.knots) - bs.spline_basis.order - 1 - 1] # 0-based
        end
        delta = x - k_pivot
        if (ders == 0)
            vec = basis(bs.spline_basis, k_pivot, 0) +
                  basis(bs.spline_basis, k_pivot, 1) * delta +
                  basis(bs.spline_basis, k_pivot, 2) * delta * delta / T(2.0) +
                  basis(bs.spline_basis, k_pivot, 3) * delta * delta * delta / T(6.0)
        elseif (ders == 1)
            vec = splines.basis(bs.spline_basis, k_pivot, 1) +
                  splines.basis(bs.spline_basis, k_pivot, 2) * delta +
                  splines.basis(bs.spline_basis, k_pivot, 3) * delta * delta / T(2.0)
        elseif (ders == 2)
            vec = splines.basis(bs.spline_basis, k_pivot, 2) +
                  splines.basis(bs.spline_basis, k_pivot, 3) * delta
        elseif (ders == 3)
            vec = splines.basis(bs.spline_basis, k_pivot, 3)
        else
            vec = k_pivot .* T(0)
        end
    else
        vec = basis(bs.spline_basis, x, ders)
    end
    if (!bs.intercept)
        vec = vec[2:length(vec)]
    end
    return vec
end

function basis(bs::AbstractSplineBasis{T}, x::AbstractVector{T}, ders::Int=0) where {T<:Real}
    f(xi) = basis(bs, xi, ders)
    return copy(transpose(reduce(hcat, f.(x))))
end

# utility function for processing the spline arguments
function spline_args(x::AbstractVector{T};
                     boundary_knots::Union{Tuple{T,T},Nothing}=nothing,
                     interior_knots::Union{Array{T,1},Nothing}=nothing,
                     order::Int=4,
                     intercept::Bool=false,
                     df::Int=3 + Int(intercept),
                     knots::Union{Array{T,1},Nothing}=nothing,
                     knots_offset::Int=0) where {T<:Real}
    if (interior_knots != nothing && boundary_knots != nothing)
        # pass
    elseif (knots != nothing)
        boundary_knots = extrema(knots)
        interior_knots = length(knots) == 2 ? nothing : knots[2:(length(knots) - 1)]
    else
        if (boundary_knots == nothing)
            boundary_knots = extrema(x)
        end
        iKnots = df - order + knots_offset + 1 - Int(intercept)
        if (iKnots > 0)
            p = range(T(0); length=iKnots + 2, stop=T(1))[2:(iKnots + 1)]
            index = (x .>= boundary_knots[1]) .* (x .<= boundary_knots[2])
            interior_knots = Statistics.quantile(x[index], p)
        end
    end
    return (boundary_knots, interior_knots)
end
