const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool   , ENV["USE_GPU"    ]) : true
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool   , ENV["DO_VIZ"     ]) : false
const do_save     = haskey(ENV, "DO_SAVE"    ) ? parse(Bool   , ENV["DO_SAVE"    ]) : false
const do_save_vis = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool   , ENV["DO_SAVE_VIZ"]) : false
const gpu_id      = haskey(ENV, "GPU_ID"     ) ? parse(Int    , ENV["GPU_ID"     ]) : 0
const nx          = haskey(ENV, "RESOL"      ) ? parse(Int    , ENV["RESOL"      ]) : 2048 - 1
const ny          = haskey(ENV, "RESOL"      ) ? parse(Int    , ENV["RESOL"      ]) : 2048 - 1
const vrpow       = haskey(ENV, "VISCR"      ) ? parse(Int    , ENV["VISCR"      ]) : 9
const nsub        = haskey(ENV, "NSUB"       ) ? parse(Int    , ENV["NSUB"       ]) : 1
const Re_mlt      = haskey(ENV, "REMULT"     ) ? parse(Float64, ENV["REMULT"     ]) : 1.0
const ρg0         = haskey(ENV, "RHOG0"      ) ? parse(Float64, ENV["RHOG0"      ]) : 0.0
const ρgi         = haskey(ENV, "RHOGI"      ) ? parse(Float64, ENV["RHOGI"      ]) : 0.0
const εbg         = haskey(ENV, "EBG"        ) ? parse(Float64, ENV["EBG"        ]) : 1.0
const simname     = haskey(ENV, "SIMNAME"    ) ?                ENV["SIMNAME"     ]  : ""
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT, Random

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel function compute_maxloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @maxloc(Musτ)
    return
end

@parallel function compute_iter_params!(dτ_Rho::Data.Array, Gdτ::Data.Array, Musτ::Data.Array, Vpdτ::Data.Number, Re::Data.Number, r::Data.Number, max_lxy::Data.Number)
    @all(dτ_Rho) = Vpdτ*max_lxy/Re/@all(Musτ)
    @all(Gdτ)    = Vpdτ^2/@all(dτ_Rho)/(r+2.0)
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, Gdτ::Data.Array, r::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - r*@all(Gdτ)*@all(∇V)
    return
end

macro μ_veτ()    esc(:( 1.0/(1.0/@all(Gdτ) + 1.0/@all(Mus)) )) end
macro μ_veτ_av() esc(:( 1.0/(1.0/@av(Gdτ)  + 1.0/@av(Mus))  )) end
@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, Gdτ::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = 2.0*@μ_veτ()*(@d_xa(Vx)/dx + @all(τxx)/@all(Gdτ)/2.0)
    @all(τyy) = 2.0*@μ_veτ()*(@d_ya(Vy)/dy + @all(τyy)/@all(Gdτ)/2.0)
    @all(τxy) = 2.0*@μ_veτ_av()*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/@av(Gdτ)/2.0)
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVx::Data.Array, dVy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dτ_Rho::Data.Array, ρg::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(Rx)  = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Ry)  = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @av_yi(ρg)
    @all(dVx) = @av_xi(dτ_Rho)*@all(Rx)
    @all(dVy) = @av_yi(dτ_Rho)*@all(Ry)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVx::Data.Array, dVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    return
end

@parallel_indices (iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel_indices (ix,iy) function create_inclusions!(Mus::Data.Array, ρg::Data.Array, Xi::Data.Array, Yi::Data.Array, r2::Data.Number, mus0::Data.Number, musi::Data.Number, ρg0::Data.Number, ρgi::Data.Number, dx::Data.Number, dy::Data.Number, lx::Data.Number, ly::Data.Number)
    if ix <= size(Mus,1) && iy <= size(Mus,2)
        x = (ix-1)*dx + 0.5*dx - 0.5*lx
        y = (iy-1)*dy + 0.5*dy - 0.5*ly
        isinside = false
        for idx = eachindex(Xi)
            rad2 = (x - Xi[idx])^2 + (y - Yi[idx])^2
            if rad2 < r2
                isinside = true
                break
            end
        end
        Mus[ix,iy] = isinside ? musi : mus0
        ρg[ix,iy]  = isinside ? ρgi  : ρg0
    end
    return
end

function generate_inclusions(lx,ly,nsub,ri)
    li    = min(3*nsub*ri,lx-2.5*ri)
    dx    = li/(nsub-1)
    dy    = dx*sqrt(3)/2
    if nsub == 1
        xs   = Float64[0]
        ys   = Float64[0]
        # random offset
        jitx = [0.25*lx*(2*rand()-1)]
        jity = [0.25*ly*(2*rand()-1)]
    else
        xs    = Float64[]
        ys    = Float64[]
        ox    = -li/2
        oy    = -li/2*sqrt(3)/2
        for j = 1:nsub
            for i = 1:(nsub-mod(j-1,2))
                push!(xs, ox + (i-1)*dx + dx/2*mod(j-1,2))
                push!(ys, oy + (j-1)*dy)
            end
        end
        # random offset
        jitx = 0.5*(dx-2*ri) .* (2 .* rand(length(xs)) .- 1.0)
        jity = 0.5*(dy-2*ri) .* (2 .* rand(length(xs)) .- 1.0)
        #exponenial falloff
        idx = xs.*jitx .> 0.0
        idy = ys.*jity .> 0.0
        jitx[idx] .*= exp.(-(xs[idx]/2).^2)
        jity[idy] .*= exp.(-(ys[idy]/2).^2)
    end
    # jitter
    xs .+= jitx
    ys .+= jity
    return xs,ys
end

@views function Stokes2D()
    # Set random seed for reproducibility
    Random.seed!(1855 + nsub)
    # Set CUDA device
    CUDA.device!(gpu_id)
    # Physics
    lx, ly    = 10.0, 10.0          # domain extends
    μs0       = 1.0                 # matrix viscosity
    μsi       = 10.0^(-vrpow)       # inclusion viscosity
    ri        = sqrt(lx*ly*0.005/π) # inclusion radius
    # Numerics
    iterMax   = 100max(nx,ny)       # maximum number of pseudo-transient iterations
    nout      = 200                 # error checking frequency
    ε_V       = 1e-5                # nonlinear absolute tolerence
    ε_∇V      = 1e-3                # nonlinear absolute tolerence
    CFL       = 0.9/sqrt(2)         # stability condition
    Re        = 5π*Re_mlt           # Reynolds number                     (numerical parameter #1)
    r         = 1.0                 # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = max(lx,ly)
    Vpdτ      = min(dx,dy)*CFL
    nsm       = 10
    # Array allocations
    Pt        = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    dVx       = @zeros(nx-1,ny-2)
    dVy       = @zeros(nx-2,ny-1)
    Mus       = @zeros(nx  ,ny  )
    Mus2      = @zeros(nx  ,ny  )
    Musτ      = @zeros(nx  ,ny  )
    Gdτ       = @zeros(nx  ,ny  )
    dτ_Rho    = @zeros(nx  ,ny  )
    ρg        = @zeros(nx  ,ny  )
    ρg2       = @zeros(nx  ,ny  )
    # Initial conditions
    Vx        =  zeros(nx+1,ny  )
    Vy        =  zeros(nx  ,ny+1)
    Vx        = Data.Array( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2)] )
    Vy        = Data.Array(  εbg.*[((iy-1)*dy -0.5*ly) for ix=1:size(Vy,1), iy=1:size(Vy,2)] )
    Xi,Yi     = Data.Array.(generate_inclusions(lx,ly,nsub,ri))
    @parallel create_inclusions!(Mus,ρg,Xi,Yi,ri^2,μs0,μsi,ρg0,ρgi,dx,dy,lx,ly)
    Mus2     .= Mus
    ρg2      .= ρg
    for _ = 1:nsm
        @parallel smooth!(Mus2, Mus, 1.0)
        Mus, Mus2 = Mus2, Mus
        @parallel smooth!(ρg2, ρg, 1.0)
        ρg, ρg2 = ρg2, ρg
    end
    Musτ     .= μs0
    # Time loop
    @parallel compute_iter_params!(dτ_Rho, Gdτ, Musτ, Vpdτ, Re, r, max_lxy)
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= iterMax)
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, Pt, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, Mus, Gdτ, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVx, dVy, Pt, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)
        @parallel (1:size(Vx,1)) bc_y!(Vx)
        @parallel (1:size(Vy,2)) bc_x!(Vy)
        iter += 1
        if iter % nout == 0
            Pmin, Pmax = minimum(Pt), maximum(Pt)
            Vmin, Vmax = minimum(Vy), maximum(Vy)
            norm_Rx    = norm(Rx)/(Pmax-Pmin)*lx/sqrt(length(Rx))
            norm_Ry    = norm(Ry)/(Pmax-Pmin)*lx/sqrt(length(Ry))
            norm_∇V    = norm(∇V)/(Vmax-Vmin)*lx/sqrt(length(∇V))
            err_V      = maximum([norm_Rx, norm_Ry])
            err_∇V     = norm_∇V
            push!(err_evo1, maximum([norm_Rx, norm_Ry])); push!(err_evo2,iter)
            @printf("Total steps = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, err_∇V)
        end
    end
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(iter-10)                      # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps = %d, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", iter, wtime, round(T_eff, sigdigits=2))
    # Visualisation
    if do_viz
        X, Y, Yv  = dx/2:dx:lx-dx/2, dy/2:dy:ly-dy/2, 0:dy:ly
        p1 = heatmap(X,  Y, Array(Pt)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="Pressure")
        p2 = heatmap(X, Yv, Array(Vy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="Vy")
        p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:viridis, title="log10(Ry)")
        p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        display(plot(p1, p2, p4, p5))
    end
    if do_save_vis && Re_mlt ≈ 1.0 && vrpow == 9 # workaround to avoid saving too much data
        outdir = joinpath("..", "out_visu", simname, "VISCR_$(vrpow)_NSUB_$(nsub)_REMULT_$(Re_mlt)")
        !ispath(outdir) && mkpath(outdir)
        matwrite(joinpath(outdir,"Stokes2D.mat"), Dict("Pt_2D"=> Array(Pt), "Mus_2D"=> Array(Mus), "Txy_2D"=> Array(τxy), "Txx_2D"=> Array(τxx), "Tyy_2D"=> Array(τyy), "Vx_2D"=> Array(Vx), "Vy_2D"=> Array(Vy), "dx_2D"=> dx, "dy_2D"=> dy); compress = true)
    end
    if do_save
        !ispath("../output") && mkpath("../output")
        open("out_Stokes2D_$(simname)_param.txt","a") do io
            println(io, "$(vrpow) $(nsub) $(Re_mlt) $(iter) $(err_V) $(err_∇V)")
        end
    end
    return
end

Stokes2D()
