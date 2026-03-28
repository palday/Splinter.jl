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

"""
    bs_(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for B-splines and return a function with signature
`(x:: Array{T,1}; ders :: Int = 0)` for evaluation of `ders`
derivative for the splines at `x`.

The keyword arguments include one of:
1. `df`, possibly in combination with `intercept`
2. `boundary_knots` and `interior_knots`
3. `knots`

# Arguments
- `boundary_knots :: Union{Tuple{T,T},Nothing} = nothing`: boundary knots
- `interior_knots :: Union{Array{T,1},Nothing} = nothing`: interior knots
- `order :: Int = 4`: order of the spline
- `intercept :: Bool = false`: bool for whether to include an intercept
- `df :: Int = order - 1 + Int(intercept)`: degrees of freedom
- `knots :: Union{Array{T,1}, Nothing} = nothing`: full set of knots (excluding repeats)
- `center :: Union{T,Nothing} = nothing)`: value to center the splines

# Examples
```jldoctest
julia> Splinter.bs_(collect(0.0:0.2:1.0), df=3)(collect(0.0:0.2:1.0))
6×3 Array{Float64,2}:
 0.0    0.0    0.0  
 0.384  0.096  0.008
 0.432  0.288  0.064
 0.288  0.432  0.216
 0.096  0.384  0.512
 0.0    0.0    1.0  
```
"""
function bs_(x::Array{T,1};
             boundary_knots::Union{Tuple{T,T},Nothing}=nothing,
             interior_knots::Union{Array{T,1},Nothing}=nothing,
             order::Int=4,
             intercept::Bool=false,
             df::Int=order - 1 + Int(intercept),
             knots::Union{Array{T,1},Nothing}=nothing,
             center::Union{T,Nothing}=nothing) where {T<:Real}
    (boundary_knots, interior_knots) = spline_args(x,
                                                   boundary_knots,
                                                   interior_knots;
                                                   order,
                                                   intercept,
                                                   df,
                                                   knots)
    spline = BSplineBasis(boundary_knots, interior_knots, order, intercept)
    function eval(x::Array{T,1}; ders::Int=0)
        b = basis(spline, x, ders)
        if (center != nothing && ders == 0)
            bc = basis(spline, center, ders)
            for i in 1:size(b, 1)
                b[i, :] -= bc
            end
        end
        return b
    end
    return eval
end

"""
    bs(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for B-splines. 

The keyword arguments include one of:
1. `df`, possibly in combination with `intercept`
2. `boundary_knots` and `interior_knots`
3. `knots`

# Arguments
- `boundary_knots :: Union{Tuple{T,T},Nothing} = nothing`: boundary knots
- `interior_knots :: Union{Array{T,1},Nothing} = nothing`: interior knots
- `order :: Int = 4`: order of the spline
- `intercept :: Bool = false`: bool for whether to include an intercept
- `df :: Int = order - 1 + Int(intercept)`: degrees of freedom
- `knots :: Union{Array{T,1}, Nothing} = nothing`: full set of knots (excluding repeats)
- `center :: Union{T,Nothing} = nothing)`: value to center the splines
- `ders :: Int = 0`: derivatives of the splines

# Examples
```jldoctest
julia> Splinter.bs(collect(0.0:0.2:1.0), df=3)
6×3 Array{Float64,2}:
 0.0    0.0    0.0  
 0.384  0.096  0.008
 0.432  0.288  0.064
 0.288  0.432  0.216
 0.096  0.384  0.512
 0.0    0.0    1.0  
```
"""
function bs(x::Array{T,1}; ders::Int=0, kwargs...) where {T<:Real}
    return bs_(x; kwargs...)(x; ders=ders)
end
