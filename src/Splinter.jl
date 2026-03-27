module Splinter

export ns, NSplineBasis, bs, bs_, is, is_, ms, ms_, basis

using OffsetArrays
using LinearAlgebra
using Statistics

zeroIndexedArray(a::Array{T}) where {T<:Real} = OffsetArray(a, Tuple(fill(-1, ndims(a))))

abstract type AbstractSplineBasis{T<:Real} end

include("basis.jl")
include("bspline.jl")
include("nspline.jl")
include("ispline.jl")
include("mspline.jl")

end # module
