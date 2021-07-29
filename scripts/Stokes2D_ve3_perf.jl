const USE_GPU = false#parse(Bool, ENV["USE_GPU"])
const do_viz  = true#parse(Bool, ENV["DO_VIZ"])
const do_save = false#parse(Bool, ENV["DO_SAVE"])
const do_save_viz = false#parse(Bool, ENV["DO_SAVE_VIZ"])
const nxx = 127#parse(Int, ENV["NX"])
const nyy = 127#parse(Int, ENV["NY"])
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel function compute_maxloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @maxloc(Musτ)
    return
end

macro Mu_eff() esc(:(1.0/(1.0/@all(Musτ) + 1.0/(G*dt)))) end
@parallel function compute_iter_params!(dt_Rho::Data.Array, Gdt::Data.Array, Musτ::Data.Array, Vpdt::Data.Number, G::Data.Number, dt::Data.Number, Re::Data.Number, r::Data.Number, max_lxy::Data.Number)
    @all(dt_Rho) = Vpdt*max_lxy/Re/@Mu_eff()
    @all(Gdt)    = Vpdt^2/@all(dt_Rho)/(r+2)
    return
end

@parallel_indices (ix,iy) function assign_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τxy_o::Data.Array)
    if (ix<=size(τxx,1) && iy<=size(τxx,2))  τxx_o[ix,iy] = τxx[ix,iy]  end
    if (ix<=size(τyy,1) && iy<=size(τyy,2))  τyy_o[ix,iy] = τyy[ix,iy]  end
    if (ix<=size(τxy,1) && iy<=size(τxy,2))  τxy_o[ix,iy] = τxy[ix,iy]  end
    return
end

macro av_Gdt(ix,iy) esc(:( 0.25*(Gdt[$ix,$iy]+Gdt[$ix+1,$iy]+Gdt[$ix,$iy+1]+Gdt[$ix+1,$iy+1]) )) end
macro Gr(ix,iy)     esc(:( Gdt[$ix,$iy]/(G*dt) )) end
macro av_Gr(ix,iy)  esc(:( @av_Gdt($ix,$iy)/(G*dt) )) end
macro av_Mus(ix,iy) esc(:( 0.25*(Mus[$ix,$iy]+Mus[$ix+1,$iy]+Mus[$ix,$iy+1]+Mus[$ix+1,$iy+1]) )) end
@parallel_indices (ix,iy) function compute_Pt_τ!(Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τxy_o::Data.Array, Gdt::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, r::Data.Number, G::Data.Number, dt::Data.Number, _dx::Data.Number, _dy::Data.Number)
    if (ix<=size(Pt,1)  && iy<=size(Pt,2))    Pt[ix,iy] = Pt[ix,iy] - r*Gdt[ix,iy]*( _dx*(Vx[ix+1,iy] - Vx[ix,iy]) + _dy*(Vy[ix,iy+1] - Vy[ix,iy]) )  end
    if (ix<=size(τxx,1) && iy<=size(τxx,2))  τxx[ix,iy] = (τxx[ix,iy] + τxx_o[ix,iy]*   @Gr(ix,iy) + 2.0*Gdt[ix,iy]*(_dx*(Vx[ix+1,iy] - Vx[ix,iy]))) / (1.0 + Gdt[ix,iy]/Mus[ix,iy] + @Gr(ix,iy))  end
    if (ix<=size(τyy,1) && iy<=size(τyy,2))  τyy[ix,iy] = (τyy[ix,iy] + τyy_o[ix,iy]*   @Gr(ix,iy) + 2.0*Gdt[ix,iy]*(_dy*(Vy[ix,iy+1] - Vy[ix,iy]))) / (1.0 + Gdt[ix,iy]/Mus[ix,iy] + @Gr(ix,iy))  end
    if (ix<=size(τxy,1) && iy<=size(τxy,2))  τxy[ix,iy] = (τxy[ix,iy] + τxy_o[ix,iy]*@av_Gr(ix,iy) + 2.0*@av_Gdt(ix,iy) * 0.5*(_dy*(Vx[ix+1,iy+1] - Vx[ix+1,iy]) + _dx*(Vy[ix+1,iy+1] - Vy[ix,iy+1])) ) / (1.0 + @av_Gdt(ix,iy)/@av_Mus(ix,iy) + @av_Gr(ix,iy))  end
    return
end

@parallel_indices (ix,iy) function compute_V!(Vx::Data.Array, Vy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dt_Rho::Data.Array, _dx::Data.Number, _dy::Data.Number, size_innVx_1, size_innVx_2, size_innVy_1, size_innVy_2)
    if (ix<=size_innVx_1 && iy<=size_innVx_2)  Vx[ix+1,iy+1] = Vx[ix+1,iy+1] + (_dx*(τxx[ix+1,iy+1] - τxx[ix,iy+1]) + _dy*(τxy[ix,iy+1] - τxy[ix,iy]) - _dx*(Pt[ix+1,iy+1] - Pt[ix,iy+1])) * (0.5(dt_Rho[ix,iy+1] + dt_Rho[ix+1,iy+1]))  end
    if (ix<=size_innVy_1 && iy<=size_innVy_2)  Vy[ix+1,iy+1] = Vy[ix+1,iy+1] + (_dy*(τyy[ix+1,iy+1] - τyy[ix+1,iy]) + _dx*(τxy[ix+1,iy] - τxy[ix,iy]) - _dy*(Pt[ix+1,iy+1] - Pt[ix+1,iy])) * (0.5(dt_Rho[ix+1,iy] + dt_Rho[ix+1,iy+1]))  end
    return
end

@parallel_indices (ix,iy) function compute_Res!(∇V::Data.Array, Rx::Data.Array, Ry::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, _dx::Data.Number, _dy::Data.Number)
    if (ix<=size(∇V,1) && iy<=size(∇V,2))  ∇V[ix,iy] = _dx*(Vx[ix+1,iy] - Vx[ix,iy]) + _dy*(Vy[ix,iy+1] - Vy[ix,iy])  end
    if (ix<=size(Rx,1) && iy<=size(Rx,2))  Rx[ix,iy] = _dx*(τxx[ix+1,iy+1] - τxx[ix,iy+1]) + _dy*(τxy[ix,iy+1] - τxy[ix,iy]) - _dx*(Pt[ix+1,iy+1] - Pt[ix,iy+1])  end
    if (ix<=size(Ry,1) && iy<=size(Ry,2))  Ry[ix,iy] = _dy*(τyy[ix+1,iy+1] - τyy[ix+1,iy]) + _dx*(τxy[ix+1,iy] - τxy[ix,iy]) - _dy*(Pt[ix+1,iy+1] - Pt[ix+1,iy])  end
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

@views function Stokes2D()
    # Physics
    lx, ly    = 10.0, 10.0  # domain extends
    ξ         = 1.0         # Maxwell relaxation time
    μs0       = 1.0         # matrix viscosity
    μsi       = 1e-3        # inclusion viscosity
    G         = 1.0         # elastic shear modulus
    εbg       = 1.0         # background strain-rate
    dt        = μs0/(G*ξ)
    # Numerics
    # nx, ny    = 1*128-1, 1*128-1    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    BLOCKX, BLOCKY = 32, 8
    GRIDX, GRIDY   = cld(nxx, BLOCKX), cld(nyy, BLOCKY)
    nx, ny  = BLOCKX*GRIDX -1, BLOCKY*GRIDY -1 # number of grid points
    @assert (nx, ny) == (nxx, nyy)
    nt        = 5           # number of time steps
    iterMax   = 2e5         # maximum number of pseudo-transient iterations
    nout      = 500         # error checking frequency
    Re        = 5π          # Reynolds number
    r         = 1.0         # Bulk to shear elastic modulus ratio
    CFL       = 0.8/sqrt(2) # CFL number # DEBUG was 0.9
    ε         = 1e-8        # nonlinear absolute tolerence
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = max(lx,ly)
    Vpdt      = min(dx,dy)*CFL
    cuthreads = (BLOCKX, BLOCKY, 1)
    cublocks  = (GRIDX,  GRIDY,  1)
    _dx, _dy  = 1.0/dx, 1.0/dy
    # Array allocations
    Pt        = @zeros(nx  ,ny  )
    dt_Rho    = @zeros(nx  ,ny  )
    Gdt       = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    τxx_o     = @zeros(nx  ,ny  )
    τyy_o     = @zeros(nx  ,ny  )
    τxy_o     = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
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
        @parallel cublocks cuthreads smooth!(Mus2, Mus, 1.0)
        Mus, Mus2 = Mus2, Mus
    end
    Musτ     .= Mus
    @parallel cublocks cuthreads compute_maxloc!(Musτ, Mus)
    @parallel (1:size(Musτ,2)) bc_x!(Musτ)
    @parallel (1:size(Musτ,1)) bc_y!(Musτ)
    size_innVx_1, size_innVx_2 = size(Vx,1)-2, size(Vx,2)-2
    size_innVy_1, size_innVy_2 = size(Vy,1)-2, size(Vy,2)-2
    # Time loop
    @parallel cublocks cuthreads compute_iter_params!(dt_Rho, Gdt, Musτ, Vpdt, G, dt, Re, r, max_lxy)
    t=0.0; ittot=0; evo_t=[]; evo_τyy=[]; err_evo1=[]; err_evo2=[]; t_tic = 0.0
    for it = 1:nt
        err=2*ε; iter=0
        @parallel cublocks cuthreads assign_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o)
        # Pseudo-transient iteration
        while err > ε && iter <= iterMax
            if (it==1 && iter==11) t_tic = Base.time()  end
            @parallel cublocks cuthreads compute_Pt_τ!(Pt, τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdt, Vx, Vy, Mus, r, G, dt, _dx, _dy)
            @parallel cublocks cuthreads compute_V!(Vx, Vy, Pt, τxx, τyy, τxy, dt_Rho, _dx, _dy, size_innVx_1, size_innVx_2, size_innVy_1, size_innVy_2)
            @parallel (1:size(Vx,1)) bc_y!(Vx)
            @parallel (1:size(Vy,2)) bc_x!(Vy)
            iter += 1
            if iter % nout == 0
                @parallel compute_Res!(∇V, Rx, Ry, Pt, Vx, Vy, τxx, τyy, τxy, _dx, _dy)
                norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
                err = maximum([norm_Rx, norm_Ry, norm_∇V])
                if isnan(err) error("NaN") end
                push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter)
                @printf("Step = %d, iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry, norm_∇V)
            end
        end
        ittot += iter; t += dt
        push!(evo_t, t); push!(evo_τyy, maximum(τyy))
    end
    # Performance
    t_toc    = Base.time() - t_tic
    A_eff    = (6*2 + 6)/1e9*nx*ny*sizeof(Data.Number) # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    t_it     = t_toc/(ittot-10)                        # Execution time per iteration [s]
    T_eff    = A_eff/t_it                              # Effective memory throughput [GB/s]
    @printf("Total iters = %d (%d steps), time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", ittot, nt, t_toc, round(T_eff, sigdigits=2))
    # Visualisation
    if do_viz
        X, Y, Yv  = dx/2:dx:lx-dx/2, dy/2:dy:ly-dy/2, 0:dy:ly
        p1 = heatmap(X,  Y, Array(τyy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="τyy")
        p2 = heatmap(X, Yv, Array(Vy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="Vy")
        p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:viridis, title="log10(Ry)")
        p5 = scatter(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, framestyle=:box, labels="max(error)", yaxis=:log10)
        p3 = plot(evo_t, evo_τyy , legend=false, xlabel="time", ylabel="max(τyy)", linewidth=0, markershape=:circle, framestyle=:box, markersize=3)
            #plot!(evo_t, 2.0.*εbg.*μs0.*(1.0.-exp.(.-evo_t.*G./μs0)), linewidth=2.0) # analytical solution
        display(plot(p1, p2, p4, p5, p3))
    end
    if do_save
        !ispath("../output") && mkdir("../output")
        open("../output/out_Stokes2D_ve3.txt","a") do io
            println(io, "$(nx) $(ny) $(ittot) $(nt)")
        end
    end
    if do_save_viz
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/Stokes_2D_ve3.mat", Dict("Pt_2D"=> Array(Pt), "Mus_2D"=> Array(Mus), "Txy_2D"=> Array(τxy), "Vy_2D"=> Array(Vy), "dx_2D"=> dx, "dy_2D"=> dy); compress = true)
    end
    return
end

Stokes2D()
