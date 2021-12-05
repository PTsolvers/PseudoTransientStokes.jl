using Test, ReferenceTests, BSON
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil

ENV["USE_GPU"] = false
ENV["DO_VIZ"]  = false
ENV["DO_SAVE"] = false
ENV["DO_SAVE_VIZ"] = false
ENV["NX"] = 63
ENV["NY"] = 63
ENV["USE_RETURN"] = true

# Reference test using ReferenceTests.jl
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2) for (v1,v2) in zip(values(d1), values(d2))])

## 2D tests
include("../scripts/Stokes2D.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d1  = Dict(:X=> xc[indsx], :P=>P[indsx,indsy])

include("../scripts/Stokes2D_ve.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d2  = Dict(:X=> xc[indsx], :P=>P[indsx,indsy])

include("../scripts/Stokes2D_ve_perf.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
d2d3  = Dict(:X=> xc[indsx], :P=>P[indsx,indsy])

@testset "Reference-tests Stokes 2D" begin
    @test_reference "reftest-files/test_Stokes_2D_v.bson" d2d1 by=comp
    @test_reference "reftest-files/test_Stokes_2D_ve.bson" d2d2 by=comp
    @test_reference "reftest-files/test_Stokes_2D_ve_perf.bson" d2d3 by=comp
end

@reset_parallel_stencil()
