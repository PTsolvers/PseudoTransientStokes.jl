using DataStructures
params = OrderedDict(
    "NSUB"   => 1:1:3,
)
static_params = Dict(
    "USE_GPU"     => false,
    "GPU_ID"      => 1,
    "DO_VIZ"      => true,
    "DO_SAVE"     => true,
    "DO_SAVE_VIZ" => false,
    "RESOL"       => 127,
    "SIMNAME"     => "aspect_ratio"
)
par_names = Iterators.flatten([typeof(par)<:Tuple ? [par...] : [par] for par ∈ keys(params)])
for par in Iterators.product(values(params)...)
    par_values = Iterators.flatten(par)
    println(collect(par_values))
    for (p,v) ∈ zip(par_names,par_values) ENV[p] = v end
    for (p,v) ∈ static_params ENV[p] = v end
    run(`julia --project -O3 --check-bounds=no Stokes2D_ve_param.jl`)
end
