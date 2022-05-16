const use_return  = haskey(ENV, "USE_RETURN" ) ? parse(Bool, ENV["USE_RETURN"] ) : false
const USE_GPU     = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : false
const do_viz      = haskey(ENV, "DO_VIZ"     ) ? parse(Bool, ENV["DO_VIZ"]     ) : true
const do_save     = haskey(ENV, "DO_SAVE"    ) ? parse(Bool, ENV["DO_SAVE"]    ) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx          = haskey(ENV, "NX"         ) ? parse(Int , ENV["NX"]         ) : 64 - 1
const ny          = haskey(ENV, "NY"         ) ? parse(Int , ENV["NY"]         ) : 64 - 1
const nz          = haskey(ENV, "NZ"         ) ? parse(Int , ENV["NZ"]         ) : 64 - 1
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics, LinearAlgebra, MAT
import MPI
# Global reductions
mean_g(A)    = (mean_l = mean(A); MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD)/MPI.Comm_size(MPI.COMM_WORLD))
norm_g(A)    = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))
maximum_g(A) = (max_l  = maximum(A); MPI.Allreduce(max_l,  MPI.MAX, MPI.COMM_WORLD))
minimum_g(A) = (min_l  = minimum(A); MPI.Allreduce(min_l,  MPI.MIN, MPI.COMM_WORLD))
# CPU functions
@views av_xza(A) = (A[1:end-1,:,1:end-1] .+ A[1:end-1,:,2:end] .+ A[2:end,:,1:end-1] .+ A[2:end,:,2:end]).*0.25
@views av_zi(A)  = (A[2:end-1,2:end-1,2:end-2] .+ A[2:end-1,2:end-1,3:end-1]).*0.5
@views av_za(A)  = (A[:,:,1:end-1] .+ A[:,:,2:end]).*0.5
@views inn(A)    =  A[2:end-1,2:end-1,2:end-1]

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2)   = @inn(A) + 1.0/6.1/fact*(@d2_xi(A) + @d2_yi(A) + @d2_zi(A))
    return
end

@parallel function compute_maxloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @maxloc(Musτ)
    return
end

macro Mu_eff() esc(:(1.0/(1.0/@all(Musτ) + 1.0/(G*dt)))) end
@parallel function compute_iter_params!(dτ_Rho::Data.Array, Gdτ::Data.Array, Musτ::Data.Array, Vpdτ::Data.Number, G::Data.Number, dt::Data.Number, Re::Data.Number, r::Data.Number, max_lxyz::Data.Number)
    @all(dτ_Rho) = Vpdτ*max_lxyz/Re/@Mu_eff()
    @all(Gdτ)    = Vpdτ^2/@all(dτ_Rho)/(r+2)
    return
end

@parallel function assign_τ!(τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array,  τxz::Data.Array,  τyz::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τzz_o::Data.Array, τxy_o::Data.Array, τxz_o::Data.Array, τyz_o::Data.Array)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τzz_o) = @all(τzz)
    @all(τxy_o) = @all(τxy)
    @all(τxz_o) = @all(τxz)
    @all(τyz_o) = @all(τyz)
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Gdτ::Data.Array, r::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
    @all(Pt)  = @all(Pt) - r*@all(Gdτ)*@all(∇V)
    return
end

macro inn_yz_Gr() esc(:( @inn_yz(Gdτ)/(G*dt) )) end
macro inn_xz_Gr() esc(:( @inn_xz(Gdτ)/(G*dt) )) end
macro inn_xy_Gr() esc(:( @inn_xy(Gdτ)/(G*dt) )) end
macro av_xyi_Gr() esc(:( @av_xyi(Gdτ)/(G*dt) )) end
macro av_xzi_Gr() esc(:( @av_xzi(Gdτ)/(G*dt) )) end
macro av_yzi_Gr() esc(:( @av_yzi(Gdτ)/(G*dt) )) end
@parallel function compute_τ!(τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τzz_o::Data.Array, τxy_o::Data.Array, τxz_o::Data.Array, τyz_o::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Mus::Data.Array, Gdτ::Data.Array, G::Data.Number, dt::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(τxx) = (@all(τxx) + @all(τxx_o)*@inn_yz_Gr() + 2.0*@inn_yz(Gdτ)*(@d_xi(Vx)/dx))/(1.0 + @inn_yz(Gdτ)/@inn_yz(Mus) + @inn_yz_Gr())
    @all(τyy) = (@all(τyy) + @all(τyy_o)*@inn_xz_Gr() + 2.0*@inn_xz(Gdτ)*(@d_yi(Vy)/dy))/(1.0 + @inn_xz(Gdτ)/@inn_xz(Mus) + @inn_xz_Gr())
    @all(τzz) = (@all(τzz) + @all(τzz_o)*@inn_xy_Gr() + 2.0*@inn_xy(Gdτ)*(@d_zi(Vz)/dz))/(1.0 + @inn_xy(Gdτ)/@inn_xy(Mus) + @inn_xy_Gr())
    @all(τxy) = (@all(τxy) + @all(τxy_o)*@av_xyi_Gr() + 2.0*@av_xyi(Gdτ)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))) /(1.0 + @av_xyi(Gdτ)/@av_xyi(Mus) + @av_xyi_Gr())
    @all(τxz) = (@all(τxz) + @all(τxz_o)*@av_xzi_Gr() + 2.0*@av_xzi(Gdτ)*(0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx))) /(1.0 + @av_xzi(Gdτ)/@av_xzi(Mus) + @av_xzi_Gr())
    @all(τyz) = (@all(τyz) + @all(τyz_o)*@av_yzi_Gr() + 2.0*@av_yzi(Gdτ)*(0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy))) /(1.0 + @av_yzi(Gdτ)/@av_yzi(Mus) + @av_yzi_Gr())
    return
end

@parallel function compute_dV!(dVx::Data.Array, dVy::Data.Array, dVz::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, dτ_Rho::Data.Array, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(dVx) = (@d_xa(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx)*@av_xi(dτ_Rho)
    @all(dVy) = (@d_ya(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy)*@av_yi(dτ_Rho)
    @all(dVz) = (@d_za(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz)*@av_zi(dτ_Rho)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dVx::Data.Array, dVy::Data.Array, dVz::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    @inn(Vz) = @inn(Vz) + @all(dVz)
    return
end

@parallel function compute_Res!(Rx::Data.Array, Ry::Data.Array, Rz::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(Rx)    = @d_xa(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx
    @all(Ry)    = @d_ya(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy
    @all(Rz)    = @d_za(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz
    return
end

@parallel_indices (iy,iz) function bc_x!(A::Data.Array)
    A[  1, iy,  iz] = A[    2,   iy,   iz]
    A[end, iy,  iz] = A[end-1,   iy,   iz]
    return
end

@parallel_indices (ix,iz) function bc_y!(A::Data.Array)
    A[ ix,  1,  iz] = A[   ix,    2,   iz]
    A[ ix,end,  iz] = A[   ix,end-1,   iz]
    return
end

@parallel_indices (ix,iy) function bc_z!(A::Data.Array)
    A[ ix,  iy,  1] = A[   ix,   iy,    2]
    A[ ix,  iy,end] = A[   ix,   iy,end-1]
    return
end

@views function Stokes3D_()
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0  # domain extends
    ξ          = 1.0               # Maxwell relaxation time
    μs0        = 1.0               # matrix viscosity
    μsi        = 1e-3              # inclusion viscosity
    G          = 1.0               # elastic shear modulus
    εbg        = 1.0               # background strain-rate
    dt         = μs0/(G*ξ)
    # Numerics
    nt         = 5                 # number of time steps
    iterMax    = 2e5               # maximum number of pseudo-transient iterations
    nout       = 200               # error checking frequency
    Re         = 6.0π              # Reynolds number
    r          = 1.0               # Bulk to shear elastic modulus ratio
    CFL        = 0.8/sqrt(3)       # CFL number
    ε          = 1e-8              # nonlinear absolute tolerence
    # nx, ny, nz = 127, 127, 127     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    b_width    = (8, 4, 4)         # boundary width for comm/comp overlap
    # Derived numerics
    me, dims   = init_global_grid(nx, ny, nz) # MPI initialisation
    dx, dy, dz = lx/nx_g(), ly/ny_g(), lz/nz_g()      # cell sizes
    max_lxyz   = max(lx,ly,lz)
    Vpdτ       = min(dx,dy,dz)*CFL
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocations
    Pt         = @zeros(nx  ,ny  ,nz  )
    RhoG       = @zeros(nx  ,ny  ,nz  )
    dτ_Rho     = @zeros(nx  ,ny  ,nz  )
    Gdτ        = @zeros(nx  ,ny  ,nz  )
    ∇V         = @zeros(nx  ,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    τxx        = @zeros(nx  ,ny-2,nz-2)
    τyy        = @zeros(nx-2,ny  ,nz-2)
    τzz        = @zeros(nx-2,ny-2,nz  )
    τxy        = @zeros(nx-1,ny-1,nz-2)
    τxz        = @zeros(nx-1,ny-2,nz-1)
    τyz        = @zeros(nx-2,ny-1,nz-1)
    τxx_o      = @zeros(nx  ,ny-2,nz-2)
    τyy_o      = @zeros(nx-2,ny  ,nz-2)
    τzz_o      = @zeros(nx-2,ny-2,nz  )
    τxy_o      = @zeros(nx-1,ny-1,nz-2)
    τxz_o      = @zeros(nx-1,ny-2,nz-1)
    τyz_o      = @zeros(nx-2,ny-1,nz-1)
    dVx        = @zeros(nx-1,ny-2,nz-2)
    dVy        = @zeros(nx-2,ny-1,nz-2)
    dVz        = @zeros(nx-2,ny-2,nz-1)
    Rx         = @zeros(nx-1,ny-2,nz-2)
    Ry         = @zeros(nx-2,ny-1,nz-2)
    Rz         = @zeros(nx-2,ny-2,nz-1)
    Mus2       = @zeros(nx  ,ny  ,nz  )
    Musτ       = @zeros(nx  ,ny  ,nz  )
    # Initial conditions
    Rad2       =  zeros(nx  ,ny  ,nz  )
    Vx         =  zeros(nx+1,ny  ,nz  )
    Vz         =  zeros(nx  ,ny  ,nz+1)
    Rad2      .= [(x_g(ix,dx,Rad2) +0.5*dx -0.5*lx)^2 + (y_g(iy,dy,Rad2) +0.5*dy -0.5*ly)^2 + (z_g(iz,dz,Rad2) +0.5*dz -0.5*lz)^2 for ix=1:size(Rad2,1), iy=1:size(Rad2,2), iz=1:size(Rad2,3)]
    Vx         = Data.Array( -εbg.*[(x_g(ix,dx,Vx) +0.5*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)] )
    Vz         = Data.Array(  εbg.*[(z_g(iz,dz,Vz) +0.5*dz -0.5*lz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)] )
    Mus        = μs0*ones(nx,ny,nz)    
    Mus[Rad2.<1.0] .= μsi
    Mus        = Data.Array( Mus )
    Mus2      .= Mus
    for ism=1:10#15
        @hide_communication b_width begin # communication/computation overlap
            @parallel smooth!(Mus2, Mus, 1.0)
            Mus, Mus2 = Mus2, Mus
            update_halo!(Mus)
        end
    end
    Musτ      .= Mus
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(Musτ, Mus)
        @parallel (1:size(Musτ,2), 1:size(Musτ,3)) bc_x!(Musτ)
        @parallel (1:size(Musτ,1), 1:size(Musτ,3)) bc_y!(Musτ)
        @parallel (1:size(Musτ,1), 1:size(Musτ,2)) bc_z!(Musτ)
        update_halo!(Musτ)
    end
    len_Rx_g   = ((nx-2-1)*dims[1]+2)*((ny-2-2)*dims[2]+2)*((nz-2-2)*dims[3]+2)
    len_Ry_g   = ((nx-2-2)*dims[1]+2)*((ny-2-1)*dims[2]+2)*((nz-2-2)*dims[3]+2)
    len_Rz_g   = ((nx-2-2)*dims[1]+2)*((ny-2-2)*dims[2]+2)*((nz-2-1)*dims[3]+2)
    len_∇V_g   = ((nx-2  )*dims[1]+2)*((ny-2  )*dims[2]+2)*((nz-2  )*dims[3]+2)
    # Preparation of visualisation
    if do_viz || do_save_viz
        if (me==0) ENV["GKSwstype"]="nul"; if do_viz !ispath("../../figures") && mkdir("../../figures") end; end
        nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        Pt_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        Vz_v   = zeros(nx_v, ny_v, nz_v)
        Rz_v   = zeros(nx_v, ny_v, nz_v)
        Mus_v  = zeros(nx_v, ny_v, nz_v)
        τxz_v  = zeros(nx_v, ny_v, nz_v)
        Pt_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        Vz_inn = zeros(nx-2, ny-2, nz-2)
        Rz_inn = zeros(nx-2, ny-2, nz-2)
        Mus_inn= zeros(nx-2, ny-2, nz-2)
        τxz_inn= zeros(nx-2, ny-2, nz-2)
        y_sl2, y_sl = Int(ceil((ny_g()-2)/2)), Int(ceil(ny_g()/2))
        Xi_g, Zi_g  = dx+dx/2:dx:(lx-dx-dx/2), dz+dz/2:dz:(lz-dz-dz/2) # inner points only
    end
    # Time loop
    @parallel compute_iter_params!(dτ_Rho, Gdτ, Musτ, Vpdτ, G, dt, Re, r, max_lxyz)
    t=0.0; ittot=0; evo_t=[]; evo_τzz=[]
    for it = 1:nt
        err=2*ε; iter=0; err_evo1=[]; err_evo2=[]
        @parallel assign_τ!(τxx, τyy, τzz, τxy, τxz, τyz, τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o)
        # Pseudo-transient iteration
        while err > ε && iter <= iterMax
            if (it==1 && iter==11)  tic()  end
            @parallel compute_P!(∇V, Pt, Vx, Vy, Vz, Gdτ, r, dx, dy, dz)
            @parallel compute_τ!(τxx, τyy, τzz, τxy, τxz, τyz, τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, Vx, Vy, Vz, Mus, Gdτ, G, dt, dx, dy, dz)
            @parallel compute_dV!(dVx, dVy, dVz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, dτ_Rho, dx, dy, dz)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(Vx, Vy, Vz, dVx, dVy, dVz)
                @parallel (1:size(Vy,2), 1:size(Vy,3)) bc_x!(Vy)
                @parallel (1:size(Vz,2), 1:size(Vz,3)) bc_x!(Vz)
                @parallel (1:size(Vx,1), 1:size(Vx,3)) bc_y!(Vx)
                @parallel (1:size(Vz,1), 1:size(Vz,3)) bc_y!(Vz)
                @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_z!(Vx)
                @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_z!(Vy)
                update_halo!(Vx, Vy, Vz)
            end
            iter += 1
            if iter % nout == 0
                @parallel compute_Res!(Rx, Ry, Rz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, dx, dy, dz)
                Vmin, Vmax = minimum_g(Vx), maximum_g(Vx)
                Pmin, Pmax = minimum_g(Pt), maximum_g(Pt)
                norm_Rx    = norm_g(Rx)/(Pmax-Pmin)*lx/sqrt(len_Rx_g)
                norm_Ry    = norm_g(Ry)/(Pmax-Pmin)*lx/sqrt(len_Ry_g)
                norm_Rz    = norm_g(Rz)/(Pmax-Pmin)*lx/sqrt(len_Rz_g)
                norm_∇V    = norm_g(∇V)/(Vmax-Vmin)*lx/sqrt(len_∇V_g)
                # norm_Rx = norm_g(Rx)/len_Rx_g; norm_Ry = norm_g(Ry)/len_Ry_g; norm_Rz = norm_g(Rz)/len_Rz_g; norm_∇V = norm_g(∇V)/len_∇V_g
                err = maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])
                if isnan(err) error("NaN") end
                push!(err_evo1,maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])); push!(err_evo2,iter)
                if (me==0) @printf("Step = %d, iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry, norm_Rz, norm_∇V) end
            end
        end
        ittot += iter; t += dt
        push!(evo_t, t); push!(evo_τzz, maximum(τzz))
    end
    # Performance
    wtime    = toc()
    A_eff    = (10*2 +1)/1e9*nx*ny*nz*sizeof(Data.Number) # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(ittot-10)                           # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                             # Effective memory throughput [GB/s]
    if (me==0) @printf("Total iters = %d (%d steps), time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", ittot, nt, wtime, round(T_eff, sigdigits=2)) end
    # Visualisation
    if do_viz || do_save_viz
        Pt_inn .= Array(inn(Pt));   gather!(Pt_inn, Pt_v)
        Vz_inn .= Array(av_zi(Vz)); gather!(Vz_inn, Vz_v)
        Rz_inn .= Array(av_za(Rz)); gather!(Rz_inn, Rz_v)
        Mus_inn.= Array(inn(Mus));  gather!(Mus_inn, Mus_v)
        τxz_inn.= Array(av_xza(τxz)); gather!(τxz_inn, τxz_v)
        if me==0 && do_viz
            p1 = heatmap(Xi_g, Zi_g, Pt_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:viridis, title="Pressure")
            p2 = heatmap(Xi_g, Zi_g, Vz_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:viridis, title="Vz")
            p4 = heatmap(Xi_g, Zi_g, log10.(abs.(Rz_v[:,y_sl2,:]')), aspect_ratio=1,  xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:viridis, title="log10(Rz)")
            #p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
            p3 = plot(evo_t, evo_τzz, legend=false, xlabel="time", ylabel="max(τzz)", linewidth=0, markershape=:circle, framestyle=:box, markersize=3)
               #plot!(evo_t, 2.0.*εbg.*μs0.*(1.0.-exp.(.-evo_t.*G./μs0)), linewidth=2.0) # analytical solution
            plot(p1, p2, p4, p3)
            savefig("../../figures/Stokes_3D_ve_$(nx_g()).png")
        end
    end
    if me==0 && do_save
        !ispath("../../output") && mkdir("../../output")
        open("../../output/out_Stokes3D_ve.txt","a") do io
            println(io, "$(nx_g()) $(ny_g()) $(nz_g()) $(ittot) $(nt)")
        end
    end
    if me==0 && do_save_viz
        !ispath("../../out_visu") && mkdir("../../out_visu")
        matwrite("../../out_visu/Stokes_3D_ve.mat", Dict("Pt_3D"=> Pt_v, "Mus_3D"=> Mus_v, "Txz_3D"=> τxz_v, "Vz_3D"=> Vz_v, "dx_3D"=> dx, "dy_3D"=> dy, "dz_3D"=> dz); compress = true)
    end
    finalize_global_grid()
    return xc, yc, zc, Pt
end

if use_return
    xc, yc, zc, P = Stokes3D_();
else
    Stokes3D = begin Stokes3D_(); return; end
end
