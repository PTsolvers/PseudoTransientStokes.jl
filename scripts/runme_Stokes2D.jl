using DataStructures
params = OrderedDict(
    "VISCR"   => 1:2:9,
    "NSUB"    => 1:5,
    "REMULT"  => 0.5:0.1:1.5
)
static_params = Dict(
    "USE_GPU"     => true,
    "DO_VIZ"      => false,
    "DO_SAVE"     => true,
    "DO_SAVE_VIZ" => true,
    "RESOL"       => 1023
)
for par in Iterators.product(values(params)...)
    outdir = joinpath("..","output",join.(zip(keys(params),par),"_")...)
    println(par)
    for (i,e) ∈ enumerate(keys(params)) ENV[e] = par[i] end
    for (i,e) ∈ static_params ENV[i] = e end
    ENV["OUTDIR"] = outdir
    run(`julia --project -O3 --check-bounds=no Stokes2D_param.jl`)
end