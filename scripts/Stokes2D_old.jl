const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, JLD

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel function compute_maxloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @maxloc(Musτ)
    return
end

@parallel function compute_timesteps!(dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, Musτ::Data.Array, Vsc::Data.Number, Ptsc::Data.Number, min_dxy2::Data.Number, β_n::Data.Number, max_nxy::Int)
    @all(dτVx) = 1.0/4.1/Vsc*min_dxy2/@av_xi(Musτ)/(1.0+β_n)
    @all(dτVy) = 1.0/4.1/Vsc*min_dxy2/@av_yi(Musτ)/(1.0+β_n)
    @all(dτPt) = 4.1*(1.0+β_n)/Ptsc/max_nxy*@all(Musτ)
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, dτPt::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, β_n::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(τxx) = 2.0*@all(Mus)*(@d_xa(Vx)/dx - 1.0/3.0*@all(∇V) + β_n*@all(∇V))
    @all(τyy) = 2.0*@all(Mus)*(@d_ya(Vy)/dy - 1.0/3.0*@all(∇V) + β_n*@all(∇V))
    @all(τxy) = 2.0*@av(Mus)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dampX::Data.Number, dampY::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Rx)    = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Ry)    = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
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

@views function Stokes2D(; nx=63, ny=63, do_viz=false)
    # Physics
    lx, ly    = 10.0, 10.0  # domain extends
    μs0       = 1.0         # matrix viscosity
    μsi       = 1e-3        # inclusion viscosity
    εbg       = 1.0         # background strain-rate
    # Numerics
    iterMax   = 1e5         # maximum number of pseudo-transient iterations
    nout      = 500         # error checking frequency
    Vdmp      = 4.0         # damping paramter for the momentum equations
    β_n       = 2.0         # numerical compressibility
    Vsc       = 1.0         # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 3.0         # relaxation paramter for the pressure equation pseudo-timestep limiter
    ε         = 1e-8        # nonlinear absolute tolerence
    # nx, ny    = 1*128-1, 1*128-1    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)
    dampX     = 1.0-Vdmp/nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/ny # damping term for the y-momentum equation
    # Array allocations
    Pt        = @zeros(nx  ,ny  )
    dτPt      = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    dVxdτ     = @zeros(nx-1,ny-2)
    dVydτ     = @zeros(nx-2,ny-1)
    dτVx      = @zeros(nx-1,ny-2)
    dτVy      = @zeros(nx-2,ny-1)
    Mus2      = @zeros(nx  ,ny  )
    Musτ      = @zeros(nx  ,ny  )
    # Initial conditions
    Rad2      =  zeros(nx  ,ny  )
    Vx        =  zeros(nx+1,ny  )
    Vy        =  zeros(nx  ,ny+1)
    Rad2     .= [((ix-1)*dx +0.5*dx -0.5*lx)^2 + ((iy-1)*dy +0.5*dy -0.5*ly)^2 for ix=1:size(Rad2,1), iy=1:size(Rad2,2)]
    Vx        = Data.Array( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2)] )
    Vy        = Data.Array(  εbg.*[((iy-1)*dy -0.5*ly) for ix=1:size(Vy,1), iy=1:size(Vy,2)] )
    Mus       = μs0*ones(nx,ny)
    Mus[Rad2.<1.0] .= μsi
    Mus       = Data.Array( Mus )
    Mus2     .= Mus
    for ism=1:10
        @parallel smooth!(Mus2, Mus, 1.0)
        Mus, Mus2 = Mus2, Mus
    end
    Musτ     .= Mus
    @parallel compute_maxloc!(Musτ, Mus)
    @parallel (1:size(Musτ,2)) bc_x!(Musτ)
    @parallel (1:size(Musτ,1)) bc_y!(Musτ)
    # Time loop
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, Musτ, Vsc, Ptsc, min_dxy2, β_n, max_nxy)
    err=2*ε; iter=0; err_evo1=[]; err_evo2=[]
    while err > ε && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, Pt, Vx, Vy, dτPt, dx, dy)
        @parallel compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, Mus, β_n, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVxdτ, dVydτ, Pt, τxx, τyy, τxy, dampX, dampY, dx, dy)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
        @parallel (1:size(Vx,1)) bc_y!(Vx)
        @parallel (1:size(Vy,2)) bc_x!(Vy)
        iter += 1
        if iter % nout == 0
            norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
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
        p1 = heatmap(X,  Y, Array(Musτ)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="Pressure")
        p2 = heatmap(X, Yv, Array(Vy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="Vy")
        p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:viridis, title="log10(Ry)")
        p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        display(plot(p1, p2, p4, p5))
    end
    return nx, ny, iter
end

#resol = 511
#Stokes2D(; nx=resol, ny=resol, do_viz=true)

@views function runtests_2D(name; do_save=false)

    resol = 16 * 2 .^ (1:9)

    out = zeros(3, length(resol))
    
    for i = 1:length(resol)

        res = resol[i]

        nxx, nyy, iter = Stokes2D(; nx=res-1, ny=res-1)

        out[1,i] = nxx
        out[2,i] = nyy
        out[3,i] = iter
    end

    if do_save
        !ispath("../output") && mkdir("../output")
        save("../output/out_$(name).jld", "out", out)
    end

end

runtests_2D("Stokes_2D"; do_save=true)
