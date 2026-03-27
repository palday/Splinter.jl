
"""
    is_(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for I-splines and return a function with signature
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

# Examples
```jldoctest
julia> Splinter.is_(collect(0.0:0.2:1.0), df=3)(collect(0.0:0.2:1.0))
6×3 Array{Float64,2}:
 0.0     0.0     0.0   
 0.1808  0.0272  0.0016
 0.5248  0.1792  0.0256
 0.8208  0.4752  0.1296
 0.9728  0.8192  0.4096
 1.0     1.0     1.0   
```
"""
function is_(x::Array{T,1};
             boundary_knots::Union{Tuple{T,T},Nothing}=nothing,
             interior_knots::Union{Array{T,1},Nothing}=nothing,
             order::Int=4,
             intercept::Bool=false,
             df::Int=order + Int(intercept),
             knots::Union{Array{T,1},Nothing}=nothing) where {T<:Real}
    (boundary_knots, interior_knots) = spline_args(x; boundary_knots=boundary_knots,
                                                   interior_knots=interior_knots,
                                                   order=order + 1,
                                                   intercept=intercept, df=df, knots=knots)
    spline = BSplineBasis(boundary_knots, interior_knots, order + 1, false)
    knots = parent(spline.spline_basis.knots)
    findj(x) = interior_knots == nothing ? (order + 1) : searchsortedlast(knots, x)
    function eval(x::Array{T,1}; ders::Int=0)
        b = basis(spline, x, ders)
        (nrow, ncol) = size(b)
        for i in 1:nrow
            js = findj(x[i])
            for j in ncol:-1:1
                if j > js
                    b[i, j] = T(0)
                elseif j == ncol
                    b[i, j] = b[i, j]
                else
                    b[i, j] = b[i, j] + b[i, j + 1]
                end
            end
            for j in ncol:-1:1
                if j < js - (order + 1) && ders == 0
                    b[i, j] = T(1)
                end
            end
        end
        return intercept ? [ones(T, nrow) b] : b
    end
    return eval
end

"""
    is(x :: Array{T,1}; <keyword arguments>) where T<:Real

Calculate a basis for I-splines. 

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
- `ders :: Int = 0`: derivatives of the splines

# Examples
```jldoctest
julia> Splinter.is(collect(0.0:0.2:1.0), df=3)
6×3 Array{Float64,2}:
 0.0     0.0     0.0   
 0.1808  0.0272  0.0016
 0.5248  0.1792  0.0256
 0.8208  0.4752  0.1296
 0.9728  0.8192  0.4096
 1.0     1.0     1.0   
```
"""
function is(x::Array{T,1}; ders::Int=0, kwargs...) where {T<:Real}
    return is_(x; kwargs...)(x; ders=ders)
end
