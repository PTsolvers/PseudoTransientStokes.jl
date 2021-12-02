const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : true
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
const do_save_vis = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx      = haskey(ENV, "RESOL") ? parse(Int, ENV["RESOL"]) : 512 - 1
const ny      = haskey(ENV, "RESOL") ? parse(Int, ENV["RESOL"]) : 512 - 1
const vrpow   = haskey(ENV, "VISCR") ? parse(Int, ENV["VISCR"]) : 9
const nsub    = haskey(ENV, "NSUB") ? parse(Int, ENV["NSUB"]) : 3
const Re_mlt  = haskey(ENV, "REMULT") ? parse(Float64, ENV["REMULT"]) : 1.0
const outdir  = haskey(ENV, "OUTDIR") ? ENV["OUTDIR"] : "../output"
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

@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, Gdτ::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = (@all(τxx) + 2.0*@all(Gdτ)*@d_xa(Vx)/dx)/(@all(Gdτ)/@all(Mus) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0*@all(Gdτ)*@d_ya(Vy)/dy)/(@all(Gdτ)/@all(Mus) + 1.0)
    @all(τxy) = (@all(τxy) + 2.0*@av(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(@av(Gdτ)/@av(Mus) + 1.0)
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVx::Data.Array, dVy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dτ_Rho::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(Rx)   = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Ry)   = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy
    @all(dVx)  = @av_xi(dτ_Rho)*@all(Rx)
    @all(dVy)  = @av_yi(dτ_Rho)*@all(Ry)
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

@parallel_indices (ix,iy) function create_inclusions!(Mus::Data.Array, Xi::Data.Array, Yi::Data.Array, r2::Data.Number, mus0::Data.Number, musi::Data.Number, dx::Data.Number, dy::Data.Number, lx::Data.Number, ly::Data.Number)
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
    end
    return
end

function generate_inclusions(lx,ly,nsub,ri)
    li    = min(4*nsub*ri,lx-2.5*ri)
    dx    = li/(nsub-1)
    dy    = dx*sqrt(3)/2
    if nsub == 1
        xs   = Float64[0]
        ys   = Float64[0]
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
    Random.seed!(1234)
    # Physics
    lx, ly    = 10.0, 10.0  # domain extends
    μs0       = 1.0         # matrix viscosity
    μsi       = 10.0^(-vrpow) # inclusion viscosity
    εbg       = 1.0         # background strain-rate
    ri        = sqrt(lx*ly*0.025/π) # inclusion radius
    # Numerics
    iterMax   = 1e5         # maximum number of pseudo-transient iterations
    nout      = 500         # error checking frequency
    ε         = 1e-7        # nonlinear absolute tolerence
    CFL       = vrpow > 7 ? 0.6/sqrt(2) :
                vrpow > 3 ? 0.7/sqrt(2) : 0.8/sqrt(2)
    Re        = 5π*Re_mlt
    r         = 1.0
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = max(lx,ly)
    Vpdτ      = min(dx,dy)*CFL
    nsm       = vrpow > 7 ? 25 :
                vrpow > 5 ? 20 :
                vrpow > 3 ? 15 : 10
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
    # Initial conditions
    Vx        =  zeros(nx+1,ny  )
    Vy        =  zeros(nx  ,ny+1)
    Vx        = Data.Array( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2)] )
    Vy        = Data.Array(  εbg.*[((iy-1)*dy -0.5*ly) for ix=1:size(Vy,1), iy=1:size(Vy,2)] )
    Xi,Yi     = Data.Array.(generate_inclusions(lx,ly,nsub,ri))
    @parallel create_inclusions!(Mus,Xi,Yi,ri^2,μs0,μsi,dx,dy,lx,ly)
    Mus2     .= Mus
    for _ = 1:nsm
        @parallel smooth!(Mus2, Mus, 1.0)
        Mus, Mus2 = Mus2, Mus
    end
    Musτ     .= Mus
    @parallel compute_maxloc!(Musτ, Mus)
    @parallel (1:size(Musτ,2)) bc_x!(Musτ)
    @parallel (1:size(Musτ,1)) bc_y!(Musτ)
    # Time loop
    @parallel compute_iter_params!(dτ_Rho, Gdτ, Musτ, Vpdτ, Re, r, max_lxy)
    err=2*ε; iter=0; err_evo1=[]; err_evo2=[]
    while err > ε && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, Pt, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, Mus, Gdτ, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVx, dVy, Pt, τxx, τyy, τxy, dτ_Rho, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)
        @parallel (1:size(Vx,1)) bc_y!(Vx)
        @parallel (1:size(Vy,2)) bc_x!(Vy)
        iter += 1
        if iter % nout == 0
            norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = 0*norm(∇V)/length(∇V)
            err = maximum([norm_Rx, norm_Ry, norm_∇V])
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_∇V)
        end
    end
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(iter-10)                      # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", iter, err, wtime, round(T_eff, sigdigits=2))
    # Visualisation
    if do_viz
        X, Y, Yv  = dx/2:dx:lx-dx/2, dy/2:dy:ly-dy/2, 0:dy:ly
        p1 = heatmap(X,  Y, Array(Pt)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="Pressure")
        p2 = heatmap(X, Yv, Array(Vy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="Vy")
        p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:viridis, title="log10(Ry)")
        p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        display(plot(p1, p2, p4, p5))
    end
    if do_save_vis && Re_mlt ≈ 1.0 && vrpow == 9
        !ispath(outdir) && mkpath(outdir)
        matwrite(joinpath(outdir,"Stokes2D.mat"), Dict("Pt_2D"=> Array(Pt), "Mus_2D"=> Array(Mus), "Txy_2D"=> Array(τxy), "Vy_2D"=> Array(Vy), "dx_2D"=> dx, "dy_2D"=> dy); compress = true)
    end
    if do_save
        !ispath(outdir) && mkpath(outdir)
        open(joinpath(outdir,"out_Stokes2D.txt"),"a") do io
            println(io, "$(nx) $(ny) $(iter) $(err)")
        end
    end
    return
end

Stokes2D()
