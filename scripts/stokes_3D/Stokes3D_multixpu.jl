const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics, LinearAlgebra, JLD
import MPI
# Global reductions
mean_g(A)    = (mean_l = mean(A);  MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD)/MPI.Comm_size(MPI.COMM_WORLD))
norm_g(A)    = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))
maximum_g(A) = (max_l  = maximum(A); MPI.Allreduce(max_l,  MPI.MAX, MPI.COMM_WORLD))
# CPU functions
@views av_zi(A) = (A[2:end-1,2:end-1,2:end-2] .+ A[2:end-1,2:end-1,3:end-1]).*0.5
@views av_za(A) = (A[:,:,1:end-1] .+ A[:,:,2:end]).*0.5
@views inn(A)   =  A[2:end-1,2:end-1,2:end-1]

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2)   = @inn(A) + 1.0/6.1/fact*(@d2_xi(A) + @d2_yi(A) + @d2_zi(A))
    return
end

@parallel function compute_maxloc!(Musτ2::Data.Array, Musτ::Data.Array)
    @inn(Musτ2) = @maxloc(Musτ)
    return
end

@parallel function compute_timesteps!(dτVx::Data.Array, dτVy::Data.Array, dτVz::Data.Array, dτPt::Data.Array, Mus::Data.Array, Vsc::Data.Number, Ptsc::Data.Number, min_dxyz2::Data.Number, max_nxyz)
    @all(dτVx) = 1.0/6.1/Vsc*min_dxyz2/@av_xi(Mus)
    @all(dτVy) = 1.0/6.1/Vsc*min_dxyz2/@av_yi(Mus)
    @all(dτVz) = 1.0/6.1/Vsc*min_dxyz2/@av_zi(Mus)
    @all(dτPt) = 6.1/Ptsc/max_nxyz*@all(Mus)
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dτPt::Data.Array, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Mus::Data.Array, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(τxx) = 2.0*@inn_yz(Mus)*(@d_xi(Vx)/dx  - 1.0/3.0*@inn_yz(∇V))
    @all(τyy) = 2.0*@inn_xz(Mus)*(@d_yi(Vy)/dy  - 1.0/3.0*@inn_xz(∇V))
    @all(τzz) = 2.0*@inn_xy(Mus)*(@d_zi(Vz)/dz  - 1.0/3.0*@inn_xy(∇V))
    @all(τxy) = 2.0*@av_xyi(Mus)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    @all(τxz) = 2.0*@av_xzi(Mus)*(0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx))
    @all(τyz) = 2.0*@av_yzi(Mus)*(0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy))
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, Rz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, dampX::Data.Number, dampY::Data.Number, dampZ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(Rx)    = @d_xa(τxx)/dx + @d_ya(τxy)/dy + @d_za(τxz)/dz - @d_xi(Pt)/dx
    @all(Ry)    = @d_ya(τyy)/dy + @d_xa(τxy)/dx + @d_za(τyz)/dz - @d_yi(Pt)/dy
    @all(Rz)    = @d_za(τzz)/dz + @d_xa(τxz)/dx + @d_ya(τyz)/dy - @d_zi(Pt)/dz
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    @all(dVzdτ) = dampZ*@all(dVzdτ) + @all(Rz)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array, dτVz::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    @inn(Vz) = @inn(Vz) + @all(dτVz)*@all(dVzdτ)
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

@views function Stokes3D(; nx=63, ny=63,  nz=63, MPI_ini_fin=true, do_viz=false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0  # domain extends
    μs0        = 1.0               # matrix viscosity
    μsi        = 1e-4              # inclusion viscosity
    εbg        = 1.0               # background strain-rate
    # Numerics
    iterMax    = 6e4               # maximum number of pseudo-transient iterations
    nout       = 1000              # error checking frequency
    Vdmp       = 4.0               # damping paramter for the momentum equations
    Vsc        = 1.0               # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc       = 8.0               # relaxation paramter for the pressure equation pseudo-timestep limiter
    ε          = 1e-8              # nonlinear absolute tolerence
    # nx, ny, nz = 127, 127, 127     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    b_width    = (8, 4, 4)         # boundary width for comm/comp overlap
    # Derived numerics
    me, dims   = init_global_grid(nx, ny, nz; init_MPI=MPI_ini_fin) # MPI initialisation
    @static if USE_GPU select_device() end    # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz = lx/nx_g(), ly/ny_g(), lz/nz_g()      # cell sizes
    min_dxyz2  = min(dx,dy,dz)^2
    max_nxyz   = max(nx_g(),ny_g(),nz_g())
    dampX      = 1.0-Vdmp/nx_g()   # damping term for the x-momentum equation
    dampY      = 1.0-Vdmp/ny_g()   # damping term for the y-momentum equation
    dampZ      = 1.0-Vdmp/nz_g()   # damping term for the z-momentum equation
    # Array allocations
    Pt         = @zeros(nx  ,ny  ,nz  )
    dτPt       = @zeros(nx  ,ny  ,nz  )
    ∇V         = @zeros(nx  ,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    τxx        = @zeros(nx  ,ny-2,nz-2)
    τyy        = @zeros(nx-2,ny  ,nz-2)
    τzz        = @zeros(nx-2,ny-2,nz  )
    τxy        = @zeros(nx-1,ny-1,nz-2)
    τxz        = @zeros(nx-1,ny-2,nz-1)
    τyz        = @zeros(nx-2,ny-1,nz-1)
    Rx         = @zeros(nx-1,ny-2,nz-2)
    Ry         = @zeros(nx-2,ny-1,nz-2)
    Rz         = @zeros(nx-2,ny-2,nz-1)
    dVxdτ      = @zeros(nx-1,ny-2,nz-2)
    dVydτ      = @zeros(nx-2,ny-1,nz-2)
    dVzdτ      = @zeros(nx-2,ny-2,nz-1)
    dτVx       = @zeros(nx-1,ny-2,nz-2)
    dτVy       = @zeros(nx-2,ny-1,nz-2)
    dτVz       = @zeros(nx-2,ny-2,nz-1)
    Mus2       = @zeros(nx  ,ny  ,nz  )
    Musτ       = @zeros(nx  ,ny  ,nz  )
    # Initial conditions
    Rad2       =  zeros(nx  ,ny  ,nz  )
    Vx         =  zeros(nx+1,ny  ,nz  )
    Vz         =  zeros(nx  ,ny  ,nz+1)
    Rad2      .= [(x_g(ix,dx,Rad2) +0.5*dx -0.5*lx)^2 + (y_g(iy,dy,Rad2) +0.5*dy -0.5*ly)^2 + (z_g(iz,dz,Rad2) +0.5*dz -0.5*lz)^2 for ix=1:size(Rad2,1), iy=1:size(Rad2,2), iz=1:size(Rad2,3)]
    Vx         = Data.Array( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)] )
    Vz         = Data.Array(  εbg.*[((iz-1)*dz -0.5*lz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)] )
    Mus        = μs0*ones(nx,ny,nz)    
    Mus[Rad2.<1.0] .= μsi
    Mus        = Data.Array( Mus )
    Mus2      .= Mus
    for ism=1:15
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
    if do_viz
        if (me==0) ENV["GKSwstype"]="nul"; !ispath("../../figures") && mkdir("../../figures") end
        nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        Pt_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        Vz_v   = zeros(nx_v, ny_v, nz_v)
        Rz_v   = zeros(nx_v, ny_v, nz_v)
        Pt_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        Vz_inn = zeros(nx-2, ny-2, nz-2)
        Rz_inn = zeros(nx-2, ny-2, nz-2)
        y_sl2, y_sl = Int(ceil((ny_g()-2)/2)), Int(ceil(ny_g()/2))
        Xi_g, Zi_g  = dx+dx/2:dx:(lx-dx-dx/2), dz+dz/2:dz:(lz-dz-dz/2) # inner points only
    end
    # Time loop
    @parallel compute_timesteps!(dτVx, dτVy, dτVz, dτPt, Musτ, Vsc, Ptsc, min_dxyz2, max_nxyz)
    err=2*ε; iter=0; err_evo1=[]; err_evo2=[]
    while err > ε && iter <= iterMax
        if (iter==11)  tic()  end
        @parallel compute_P!(∇V, Pt, Vx, Vy, Vz, dτPt, dx, dy, dz)
        @parallel compute_τ!(∇V, τxx, τyy, τzz, τxy, τxz, τyz, Vx, Vy, Vz, Mus, dx, dy, dz)
        @parallel compute_dV!(Rx, Ry, Rz, dVxdτ, dVydτ, dVzdτ, Pt, τxx, τyy, τzz, τxy, τxz, τyz, dampX, dampY, dampZ, dx, dy, dz)
        @hide_communication b_width begin # communication/computation overlap
            @parallel compute_V!(Vx, Vy, Vz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz)
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
            norm_Rx = norm_g(Rx)/len_Rx_g; norm_Ry = norm_g(Ry)/len_Ry_g; norm_Rz = norm_g(Rz)/len_Rz_g; norm_∇V = norm_g(∇V)/len_∇V_g
            err = maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])
            push!(err_evo1,maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])); push!(err_evo2,iter)
            if (me==0) @printf("Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_Rz, norm_∇V) end
        end
    end
    # Performance
    wtime    = toc()
    A_eff    = (4*2)/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(iter-10)                         # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]
    if (me==0) @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", iter, err, wtime, round(T_eff, sigdigits=2)) end
    # Visualisation
    if do_viz
        Pt_inn .= inn(Pt);   gather!(Pt_inn, Pt_v)
        Vz_inn .= av_zi(Vz); gather!(Vz_inn, Vz_v)
        Rz_inn .= av_za(Rz); gather!(Rz_inn, Rz_v)
        if (me==0)
            p1 = heatmap(Xi_g, Zi_g, Pt_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:viridis, title="Pressure")
            p2 = heatmap(Xi_g, Zi_g, Vz_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:viridis, title="Vz")
            p4 = heatmap(Xi_g, Zi_g, log10.(abs.(Rz_v[:,y_sl2,:]')), aspect_ratio=1,  xlims=(Xi_g[1],Xi_g[end]), zlims=(Zi_g[1],Zi_g[end]), c=:viridis, title="log10(Rz)")
            p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
            plot(p1, p2, p4, p5)
            savefig("../../figures/Stokes_3D_$(nx_g()).png")
        end
    end
    nxg, nyg, nzg = nx_g(), ny_g(), nz_g()
    finalize_global_grid(; finalize_MPI=MPI_ini_fin)
    return nxg, nyg, nzg, iter, me
end

Stokes3D(; nx=255, ny=255, nz=255, do_viz=true)

# @views function runtests_3D(name; do_save=false)

#     resol = 16 * 2 .^ (1:5)

#     out = zeros(4, length(resol))
#     me  = 0
    
#     MPI.Init()
    
#     for i = 1:length(resol)

#         res = resol[i]

#         nxx, nyy, nzz, iter, me = Stokes3D(; nx=res-1, ny=res-1, nz=res-1, MPI_ini_fin=false)

#         out[1,i] = nxx
#         out[2,i] = nyy
#         out[3,i] = nzz
#         out[4,i] = iter
#     end

#     if do_save && me==0
#         !ispath("../../output") && mkdir("../../output")
#         save("../../output/out_$(name).jld", "out", out)
#     end

#     MPI.Finalize()
# end

# runtests_3D("Stokes_3D"; do_save=true)
