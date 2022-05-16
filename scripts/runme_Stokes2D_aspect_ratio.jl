using DataStructures
params = OrderedDict(
    # "NSUB"   => (1,2,4,8),
    # "RESOL"  => (127, 255, 511, 1023, 2047)
    "FACT"   => (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    "NSUB"   => (1, 2, 4, 8),
    "RESOL"  => (127, 255, 511, 1023, 2047)
)
static_params = Dict(
    "USE_GPU"     => true,
    "GPU_ID"      => 7,
    "DO_VIZ"      => false,
    "DO_SAVE"     => true,
    "DO_SAVE_VIZ" => false,
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
