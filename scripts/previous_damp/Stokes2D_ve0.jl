const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : true
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx = haskey(ENV, "NX") ? parse(Int, ENV["NX"]) : 512 - 1
const ny = haskey(ENV, "NY") ? parse(Int, ENV["NY"]) : 512 - 1
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

@parallel function compute_timesteps!(dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, Musτ::Data.Array, Vsc::Data.Number, Ptsc::Data.Number, min_dxy2::Data.Number, β_n::Data.Number, max_nxy::Int)
    @all(dτVx) = 1.0/4.1/Vsc*min_dxy2/@av_xi(Musτ)/(1.0+β_n)
    @all(dτVy) = 1.0/4.1/Vsc*min_dxy2/@av_yi(Musτ)/(1.0+β_n)
    @all(dτPt) = 4.1*(1.0+β_n)/Ptsc/max_nxy*@all(Musτ)
    return
end

@parallel function assign_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τxy_o::Data.Array)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, dτPt::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

macro Xsi()    esc(:( @all(Mus)/(G*dt) )) end
macro av_Xsi() esc(:(  @av(Mus)/(G*dt) )) end
@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τxy_o::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, ∇V::Data.Array, G::Data.Number, dt::Data.Number, β_n::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(τxx) = @all(τxx_o)*@Xsi()/(@Xsi()+1.0) + 2.0*@all(Mus)/(@Xsi()+1.0)*( @d_xa(Vx)/dx - 1.0/3.0*@all(∇V) + β_n*@all(∇V) )
    @all(τyy) = @all(τyy_o)*@Xsi()/(@Xsi()+1.0) + 2.0*@all(Mus)/(@Xsi()+1.0)*( @d_ya(Vy)/dy - 1.0/3.0*@all(∇V) + β_n*@all(∇V) )
    @all(τxy) = @all(τxy_o)*@av_Xsi()/(@av_Xsi()+1.0) + 2.0*@av(Mus)/(@av_Xsi()+1.0)*( 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) )
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
    τxx_o     = @zeros(nx  ,ny  )
    τyy_o     = @zeros(nx  ,ny  )
    τxy_o     = @zeros(nx-1,ny-1)
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
    t=0.0; ittot=0; evo_t=[]; evo_τyy=[]
    for it = 1:nt
        err=2*ε; iter=0; err_evo1=[]; err_evo2=[]
        @parallel assign_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o)
        # Pseudo-transient iteration
        while err > ε && iter <= iterMax
            if (it==1 && iter==11)  global wtime0 = Base.time()  end
            @parallel compute_P!(∇V, Pt, Vx, Vy, dτPt, dx, dy)
            @parallel compute_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Vx, Vy, Mus, ∇V, G, dt, β_n, dx, dy)
            @parallel compute_dV!(Rx, Ry, dVxdτ, dVydτ, Pt, τxx, τyy, τxy, dampX, dampY, dx, dy)
            @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
            @parallel (1:size(Vx,1)) bc_y!(Vx)
            @parallel (1:size(Vy,2)) bc_x!(Vy)
            iter += 1
            if iter % nout == 0
                Vmin, Vmax = minimum(Vx), maximum(Vx)
                Pmin, Pmax = minimum(Pt), maximum(Pt)
                norm_Rx    = norm(Rx)/(Pmax-Pmin)*lx/sqrt(length(Rx))
                norm_Ry    = norm(Ry)/(Pmax-Pmin)*lx/sqrt(length(Ry))
                norm_∇V    = norm(∇V)/(Vmax-Vmin)*lx/sqrt(length(∇V))
                # norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
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
        #p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        p3 = plot(evo_t, evo_τyy , legend=false, xlabel="time", ylabel="max(τyy)", linewidth=0, markershape=:circle, framestyle=:box, markersize=3)
            #plot!(evo_t, 2.0.*εbg.*μs0.*(1.0.-exp.(.-evo_t.*G./μs0)), linewidth=2.0) # analytical solution
        display(plot(p1, p2, p4, p3))
    end
    if do_save
        !ispath("../output") && mkdir("../output")
        open("../output/out_Stokes2D_ve0.txt","a") do io
            println(io, "$(nx) $(ny) $(ittot) $(nt)")
        end
    end
    if do_save_viz
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/Stokes_2D_ve0.mat", Dict("Pt_2D"=> Array(Pt), "Mus_2D"=> Array(Mus), "Txy_2D"=> Array(τxy), "Vy_2D"=> Array(Vy), "dx_2D"=> dx, "dy_2D"=> dy); compress = true)
    end
    return
end

Stokes2D()
