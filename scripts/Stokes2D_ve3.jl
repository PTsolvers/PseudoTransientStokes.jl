const USE_GPU = false #parse(Bool, ENV["USE_GPU"])
const do_viz  = true #parse(Bool, ENV["DO_VIZ"])
const do_save = false #parse(Bool, ENV["DO_SAVE"])
const nx = 127 #parse(Int, ENV["NX"])
const ny = 127 #parse(Int, ENV["NY"])
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel function compute_maxloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @maxloc(Musτ)
    return
end

macro Mu_eff() esc(:(1.0/(1.0/@all(Musτ) + 1.0/(G*dt)))) end
@parallel function compute_iter_params!(dt_Rho::Data.Array, Gdt::Data.Array, Musτ::Data.Array, Vpdt::Data.Number, G::Data.Number, dt::Data.Number, Re::Data.Number, max_lxy::Data.Number)
    @all(dt_Rho) = Vpdt*max_lxy/Re/@Mu_eff()
    @all(Gdt)    = Vpdt^2/@all(dt_Rho)
    return
end

@parallel function assign_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τxy_o::Data.Array)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Gdt::Data.Array, Vx::Data.Array, Vy::Data.Array, r::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(∇V) = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt) = @all(Pt) - r*@all(Gdt)*@all(∇V)
    return
end

macro Gr()    esc(:( @all(Gdt)/(G*dt) )) end
macro av_Gr() esc(:(  @av(Gdt)/(G*dt) )) end
@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, txx_o::Data.Array, tyy_o::Data.Array, txy_o::Data.Array, Gdt::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, G::Data.Number, dt::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(τxx) = (@all(τxx) + @all(txx_o)*   @Gr() + 2.0*@all(Gdt)*(@d_xa(Vx)/dx))/(1.0 + @all(Gdt)/@all(Mus) + @Gr())
    @all(τyy) = (@all(τyy) + @all(tyy_o)*   @Gr() + 2.0*@all(Gdt)*(@d_ya(Vy)/dy))/(1.0 + @all(Gdt)/@all(Mus) + @Gr())
    @all(τxy) = (@all(τxy) + @all(txy_o)*@av_Gr() + 2.0*@av(Gdt)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)))/(1.0 + @av(Gdt)/@av(Mus) + @av_Gr())
    return
end

@parallel function compute_dV!(dVx::Data.Array, dVy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dt_Rho::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(dVx) = (@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx)*@av_xi(dt_Rho)
    @all(dVy) = (@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy)*@av_yi(dt_Rho)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVx::Data.Array, dVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    return
end

@parallel function compute_Res!(Rx::Data.Array, Ry::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(Rx) = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Ry) = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy
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
    nt        = 5           # number of time steps
    iterMax   = 2e5         # maximum number of pseudo-transient iterations
    nout      = 200         # error checking frequency
    Re        = 3π          # Reynolds number squared
    r         = 1.0         # Bulk to shear elastic modulus ratio
    CFL       = 1/3         # CFL number
    ε         = 1e-8        # nonlinear absolute tolerence
    # nx, ny    = 1*128-1, 1*128-1    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = max(lx,ly)
    Vpdt      = min(dx,dy)*CFL
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
    dVx       = @zeros(nx-1,ny-2)
    dVy       = @zeros(nx-2,ny-1)
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
        @parallel smooth!(Mus2, Mus, 1.0)
        Mus, Mus2 = Mus2, Mus
    end
    Musτ     .= Mus
    @parallel compute_maxloc!(Musτ, Mus)
    @parallel (1:size(Musτ,2)) bc_x!(Musτ)
    @parallel (1:size(Musτ,1)) bc_y!(Musτ)
    # Time loop
    @parallel compute_iter_params!(dt_Rho, Gdt, Musτ, Vpdt, G, dt, Re, max_lxy)
    t=0.0; ittot=0; evo_t=[]; evo_τyy=[]; err_evo1=[]; err_evo2=[]
    for it = 1:nt
        err=2*ε; iter=0
        @parallel assign_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o)
        # Pseudo-transient iteration
        while err > ε && iter <= iterMax
            if (it==1 && iter==11)  global wtime0 = Base.time()  end
            @parallel compute_P!(∇V, Pt, Gdt, Vx, Vy, r, dx, dy)
            @parallel compute_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdt, Vx, Vy, Mus, G, dt, dx, dy)
            @parallel compute_dV!(dVx, dVy, Pt, τxx, τyy, τxy, dt_Rho, dx, dy)
            @parallel compute_V!(Vx, Vy, dVx, dVy)
            @parallel (1:size(Vx,1)) bc_y!(Vx)
            @parallel (1:size(Vy,2)) bc_x!(Vy)
            iter += 1
            if iter % nout == 0
                @parallel compute_Res!(Rx, Ry, Pt, τxx, τyy, τxy, dx, dy)
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
    wtime    = Base.time() - wtime0
    A_eff    = (6*2 + 1)/1e9*nx*ny*sizeof(Data.Number) # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(ittot-10)                         # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]
    @printf("Total iters = %d (%d steps), time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", ittot, nt, wtime, round(T_eff, sigdigits=2))
    # Visualisation
    if do_viz
        X, Y, Yv  = dx/2:dx:lx-dx/2, dy/2:dy:ly-dy/2, 0:dy:ly
        p1 = heatmap(X,  Y, Array(τyy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:viridis, title="τyy")
        p2 = heatmap(X, Yv, Array(Vy)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Yv[1],Yv[end]), c=:viridis, title="Vy")
        p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:viridis, title="log10(Ry)")
        p5 = scatter(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
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
    return
end

Stokes2D()
