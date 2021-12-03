
include("./shared.jl")

## 3D tests
include("../scripts/stokes_3D/Stokes3D_ve_multixpu.jl")
@reset_parallel_stencil()
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(yc), 12)))
indsz = Int.(ceil.(LinRange(1, length(zc), 12)))
d3d   = Dict(:X=> xc[indsx], :P=>P[indsx,indsy,indsz])

@testset "Reference-tests Stokes 3D VE" begin
    @test_reference "reftest-files/test_Stokes3D_ve_multixpu.bson" d3d by=comp
end

@reset_parallel_stencil()
