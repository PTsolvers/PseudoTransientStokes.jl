using DataStructures
params = OrderedDict(
    "VISCR"           => 1:4:9,
    "NSUB"            => 1:3:7,
    "REMULT"          => 0.25:0.25:1.5
)
static_params = Dict(
    "USE_GPU"     => true,
    "GPU_ID"      => 0,
    "DO_VIZ"      => false,
    "DO_SAVE"     => true,
    "DO_SAVE_VIZ" => true,
    "RESOL"       => 2047,
    "EBG"         => 0.0,
    "RHOG0"       => 1.0,
    "RHOGI"       => 0.5,
    "SIMDIR"      => "../output/gravity_weak"
)
par_names = Iterators.flatten([typeof(par)<:Tuple ? [par...] : [par] for par ∈ keys(params)])
for par in Iterators.product(values(params)...)
    par_values = Iterators.flatten(par)
    println(collect(par_values))
    for (p,v) ∈ zip(par_names,par_values) ENV[p] = v end
    for (p,v) ∈ static_params ENV[p] = v end
    run(`julia --project -O3 --check-bounds=no Stokes2D_param.jl`)
end
