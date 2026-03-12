include("set_up_tests.jl")

@testset "Aqua" begin
    Aqua.test_all(Splines2; ambiguities=false)
end

@testset "splines" include("splines.jl")
@testset "StatsModels" include("statsmodels.jl")
