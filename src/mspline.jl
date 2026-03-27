
"""
    ms_(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for M-splines and return a function with signature
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
- `intercept :: Bool = false`: bool for whether to include an intercept (column of ones). This behaviour is different to the Splinter package from R, where intercept=FALSE will drop the first spline term.
- `df :: Int = order + Int(intercept)`: degrees of freedom
- `knots :: Union{Array{T,1}, Nothing} = nothing`: full set of knots (excluding repeats)
- `center :: Union{T,Nothing} = nothing)`: value to center the splines

# Examples
```jldoctest
julia> Splinter.ms_(collect(0.0:0.2:1.0), df=3)(collect(0.0:0.2:1.0))
6×3 Array{Float64,2}:
 0.0    0.0    0.0  
 1.536  0.384  0.032
 1.728  1.152  0.256
 1.152  1.728  0.864
 0.384  1.536  2.048
 0.0    0.0    4.0  
```
"""
function ms_(x::Array{T,1};
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
        knots = parent(spline.spline_basis.knots)
        function loop(j)
            denom = knots[j + order] - knots[j]
            return denom > T(0) ? order / denom : T(0)
        end
        transCoef = map(loop, 1:(length(knots) - order))
        if (!intercept)
            transCoef = transCoef[2:length(transCoef)]
        end
        if (center != nothing && ders == 0)
            bc = basis(spline, center, ders)
            for i in 1:size(b, 1)
                b[i, :] -= bc
            end
        end
        for i in 1:size(b, 1)
            b[i, :] .*= transCoef
        end
        return b
    end
    return eval
end

"""
    ms(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for M-splines. 

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
julia> Splinter.ms(collect(0.0:0.2:1.0), df=3)
6×3 Array{Float64,2}:
 0.0    0.0    0.0  
 1.536  0.384  0.032
 1.728  1.152  0.256
 1.152  1.728  0.864
 0.384  1.536  2.048
 0.0    0.0    4.0  
```
"""
function ms(x::Array{T,1}; ders::Int=0, kwargs...) where {T<:Real}
    return ms_(x; kwargs...)(x; ders=ders)
end
