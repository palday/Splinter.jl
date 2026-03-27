module Splinter

export ns, NSplineBasis, bs, bs_, is, is_, ms, ms_, basis

using OffsetArrays
using LinearAlgebra
using Statistics

zeroArray(a::Array{T}) where {T<:Real} = OffsetArray(a, (size(a) .* 0) .- 1)

abstract type AbstractSplineBasis{T<:Real} end

include("basis.jl")
include("bspline.jl")
include("nspline.jl")
include("ispline.jl")
include("mspline.jl")


end # module
