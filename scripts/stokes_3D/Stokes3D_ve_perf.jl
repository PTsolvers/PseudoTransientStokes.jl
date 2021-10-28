const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : true
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : false
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false
const nx = haskey(ENV, "NX") ? parse(Int, ENV["NX"]) : 64 - 1
const ny = haskey(ENV, "NY") ? parse(Int, ENV["NY"]) : 64 - 1
const nz = haskey(ENV, "NZ") ? parse(Int, ENV["NZ"]) : 64 - 1
###
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT
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
@parallel function compute_iter_params!(dt_Rho::Data.Array, Gdt::Data.Array, Musτ::Data.Array, Vpdt::Data.Number, G::Data.Number, dt::Data.Number, Re::Data.Number, r::Data.Number, max_lxyz::Data.Number)
    @all(dt_Rho) = Vpdt*max_lxyz/Re/@Mu_eff()
    @all(Gdt)    = Vpdt^2/@all(dt_Rho)/(r+2)
    return
end

@parallel_indices (ix,iy,iz) function assign_τ!(τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array,  τxz::Data.Array,  τyz::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τzz_o::Data.Array, τxy_o::Data.Array, τxz_o::Data.Array, τyz_o::Data.Array)
    if (ix<=size(τxx,1) && iy<=size(τxx,2) && iz<=size(τxx,3))  τxx_o[ix,iy,iz] = τxx[ix,iy,iz]  end
    if (ix<=size(τyy,1) && iy<=size(τyy,2) && iz<=size(τyy,3))  τyy_o[ix,iy,iz] = τyy[ix,iy,iz]  end
    if (ix<=size(τzz,1) && iy<=size(τzz,2) && iz<=size(τzz,3))  τzz_o[ix,iy,iz] = τzz[ix,iy,iz]  end
    if (ix<=size(τxy,1) && iy<=size(τxy,2) && iz<=size(τxy,3))  τxy_o[ix,iy,iz] = τxy[ix,iy,iz]  end
    if (ix<=size(τxz,1) && iy<=size(τxz,2) && iz<=size(τxz,3))  τxz_o[ix,iy,iz] = τxz[ix,iy,iz]  end
    if (ix<=size(τyz,1) && iy<=size(τyz,2) && iz<=size(τyz,3))  τyz_o[ix,iy,iz] = τyz[ix,iy,iz]  end
    return
end

macro inn_yz_Gdt(ix,iy,iz) esc(:( Gdt[$ix  ,$iy+1,$iz+1] )) end
macro inn_xz_Gdt(ix,iy,iz) esc(:( Gdt[$ix+1,$iy  ,$iz+1] )) end
macro inn_xy_Gdt(ix,iy,iz) esc(:( Gdt[$ix+1,$iy+1,$iz  ] )) end
macro inn_yz_Mus(ix,iy,iz) esc(:( Mus[$ix  ,$iy+1,$iz+1] )) end
macro inn_xz_Mus(ix,iy,iz) esc(:( Mus[$ix+1,$iy  ,$iz+1] )) end
macro inn_xy_Mus(ix,iy,iz) esc(:( Mus[$ix+1,$iy+1,$iz  ] )) end
macro av_xyi_Gdt(ix,iy,iz) esc(:( (Gdt[$ix  ,$iy  ,$iz+1] + Gdt[$ix+1,$iy  ,$iz+1] + Gdt[$ix  ,$iy+1,$iz+1] + Gdt[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_xzi_Gdt(ix,iy,iz) esc(:( (Gdt[$ix  ,$iy+1,$iz  ] + Gdt[$ix+1,$iy+1,$iz  ] + Gdt[$ix  ,$iy+1,$iz+1] + Gdt[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_yzi_Gdt(ix,iy,iz) esc(:( (Gdt[$ix+1,$iy  ,$iz  ] + Gdt[$ix+1,$iy+1,$iz  ] + Gdt[$ix+1,$iy  ,$iz+1] + Gdt[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_xyi_Mus(ix,iy,iz) esc(:( (Mus[$ix  ,$iy  ,$iz+1] + Mus[$ix+1,$iy  ,$iz+1] + Mus[$ix  ,$iy+1,$iz+1] + Mus[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_xzi_Mus(ix,iy,iz) esc(:( (Mus[$ix  ,$iy+1,$iz  ] + Mus[$ix+1,$iy+1,$iz  ] + Mus[$ix  ,$iy+1,$iz+1] + Mus[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro av_yzi_Mus(ix,iy,iz) esc(:( (Mus[$ix+1,$iy  ,$iz  ] + Mus[$ix+1,$iy+1,$iz  ] + Mus[$ix+1,$iy  ,$iz+1] + Mus[$ix+1,$iy+1,$iz+1])*0.25 )) end
macro inn_yz_Gr(ix,iy,iz) esc(:( @inn_yz_Gdt($ix,$iy,$iz)/(G*dt) )) end
macro inn_xz_Gr(ix,iy,iz) esc(:( @inn_xz_Gdt($ix,$iy,$iz)/(G*dt) )) end
macro inn_xy_Gr(ix,iy,iz) esc(:( @inn_xy_Gdt($ix,$iy,$iz)/(G*dt) )) end
macro av_xyi_Gr(ix,iy,iz) esc(:( @av_xyi_Gdt($ix,$iy,$iz)/(G*dt) )) end
macro av_xzi_Gr(ix,iy,iz) esc(:( @av_xzi_Gdt($ix,$iy,$iz)/(G*dt) )) end
macro av_yzi_Gr(ix,iy,iz) esc(:( @av_yzi_Gdt($ix,$iy,$iz)/(G*dt) )) end
@parallel_indices (ix,iy,iz) function compute_Pt_τ!(Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, τxx_o::Data.Array, τyy_o::Data.Array, τzz_o::Data.Array, τxy_o::Data.Array, τxz_o::Data.Array, τyz_o::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Mus::Data.Array, Gdt::Data.Array, r::Data.Number, G::Data.Number, dt::Data.Number, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)
    if (ix<=size(Pt,1)  && iy<=size(Pt,2)  && iz<=size(Pt,3))    Pt[ix,iy,iz] = Pt[ix,iy,iz] - r*Gdt[ix,iy,iz]*(_dx*(Vx[ix+1,iy,iz] - Vx[ix,iy,iz]) + _dy*(Vy[ix,iy+1,iz] - Vy[ix,iy,iz]) + _dz*(Vz[ix,iy,iz+1] - Vz[ix,iy,iz]))  end
    if (ix<=size(τxx,1) && iy<=size(τxx,2) && iz<=size(τxx,3))  τxx[ix,iy,iz] = ( τxx[ix,iy,iz] + τxx_o[ix,iy,iz]*@inn_yz_Gr(ix,iy,iz) + 2.0*@inn_yz_Gdt(ix,iy,iz)*(_dx*(Vx[ix+1,iy+1,iz+1] - Vx[ix,iy+1,iz+1])))/(1.0 + @inn_yz_Gdt(ix,iy,iz)/@inn_yz_Mus(ix,iy,iz) + @inn_yz_Gr(ix,iy,iz))  end
    if (ix<=size(τyy,1) && iy<=size(τyy,2) && iz<=size(τyy,3))  τyy[ix,iy,iz] = ( τyy[ix,iy,iz] + τyy_o[ix,iy,iz]*@inn_xz_Gr(ix,iy,iz) + 2.0*@inn_xz_Gdt(ix,iy,iz)*(_dy*(Vy[ix+1,iy+1,iz+1] - Vy[ix+1,iy,iz+1])))/(1.0 + @inn_xz_Gdt(ix,iy,iz)/@inn_xz_Mus(ix,iy,iz) + @inn_xz_Gr(ix,iy,iz))  end
    if (ix<=size(τzz,1) && iy<=size(τzz,2) && iz<=size(τzz,3))  τzz[ix,iy,iz] = ( τzz[ix,iy,iz] + τzz_o[ix,iy,iz]*@inn_xy_Gr(ix,iy,iz) + 2.0*@inn_xy_Gdt(ix,iy,iz)*(_dz*(Vz[ix+1,iy+1,iz+1] - Vz[ix+1,iy+1,iz])))/(1.0 + @inn_xy_Gdt(ix,iy,iz)/@inn_xy_Mus(ix,iy,iz) + @inn_xy_Gr(ix,iy,iz))  end
    if (ix<=size(τxy,1) && iy<=size(τxy,2) && iz<=size(τxy,3))  τxy[ix,iy,iz] = ( τxy[ix,iy,iz] + τxy_o[ix,iy,iz]*@av_xyi_Gr(ix,iy,iz) + 2.0*@av_xyi_Gdt(ix,iy,iz)*(0.5*(_dy*(Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy,iz+1]) + _dx*(Vy[ix+1,iy+1,iz+1] - Vy[ix,iy+1,iz+1])))) / (1.0 + @av_xyi_Gdt(ix,iy,iz)/@av_xyi_Mus(ix,iy,iz) + @av_xyi_Gr(ix,iy,iz))  end
    if (ix<=size(τxz,1) && iy<=size(τxz,2) && iz<=size(τxz,3))  τxz[ix,iy,iz] = ( τxz[ix,iy,iz] + τxz_o[ix,iy,iz]*@av_xzi_Gr(ix,iy,iz) + 2.0*@av_xzi_Gdt(ix,iy,iz)*(0.5*(_dz*(Vx[ix+1,iy+1,iz+1] - Vx[ix+1,iy+1,iz]) + _dx*(Vz[ix+1,iy+1,iz+1] - Vz[ix,iy+1,iz+1])))) / (1.0 + @av_xzi_Gdt(ix,iy,iz)/@av_xzi_Mus(ix,iy,iz) + @av_xzi_Gr(ix,iy,iz))  end
    if (ix<=size(τyz,1) && iy<=size(τyz,2) && iz<=size(τyz,3))  τyz[ix,iy,iz] = ( τyz[ix,iy,iz] + τyz_o[ix,iy,iz]*@av_yzi_Gr(ix,iy,iz) + 2.0*@av_yzi_Gdt(ix,iy,iz)*(0.5*(_dz*(Vy[ix+1,iy+1,iz+1] - Vy[ix+1,iy+1,iz]) + _dy*(Vz[ix+1,iy+1,iz+1] - Vz[ix+1,iy,iz+1])))) / (1.0 + @av_yzi_Gdt(ix,iy,iz)/@av_yzi_Mus(ix,iy,iz) + @av_yzi_Gr(ix,iy,iz))  end
    return
end

macro av_xi_dt_Rho(ix,iy,iz)  esc(:( (dt_Rho[$ix  ,$iy+1,$iz+1] + dt_Rho[$ix+1,$iy+1,$iz+1])*0.5 )) end
macro av_yi_dt_Rho(ix,iy,iz)  esc(:( (dt_Rho[$ix+1,$iy  ,$iz+1] + dt_Rho[$ix+1,$iy+1,$iz+1])*0.5 )) end
macro av_zi_dt_Rho(ix,iy,iz)  esc(:( (dt_Rho[$ix+1,$iy+1,$iz  ] + dt_Rho[$ix+1,$iy+1,$iz+1])*0.5 )) end
@parallel_indices (ix,iy,iz) function compute_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, dt_Rho::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number, nx_1, nx_2, ny_1, ny_2, nz_1, nz_2)
    if (ix<=nx_1 && iy<=ny_2 && iz<=nz_2)  Vx[ix+1,iy+1,iz+1] = Vx[ix+1,iy+1,iz+1] + (_dx*(τxx[ix+1,iy,iz] - τxx[ix,iy,iz]) + _dy*(τxy[ix,iy+1,iz] - τxy[ix,iy,iz]) + _dz*(τxz[ix,iy,iz+1] - τxz[ix,iy,iz]) - _dx*(Pt[ix+1,iy+1,iz+1] - Pt[ix,iy+1,iz+1]))*@av_xi_dt_Rho(ix,iy,iz)  end
    if (ix<=nx_2 && iy<=ny_1 && iz<=nz_2)  Vy[ix+1,iy+1,iz+1] = Vy[ix+1,iy+1,iz+1] + (_dy*(τyy[ix,iy+1,iz] - τyy[ix,iy,iz]) + _dx*(τxy[ix+1,iy,iz] - τxy[ix,iy,iz]) + _dz*(τyz[ix,iy,iz+1] - τyz[ix,iy,iz]) - _dy*(Pt[ix+1,iy+1,iz+1] - Pt[ix+1,iy,iz+1]))*@av_yi_dt_Rho(ix,iy,iz)  end
    if (ix<=nx_2 && iy<=ny_2 && iz<=nz_1)  Vz[ix+1,iy+1,iz+1] = Vz[ix+1,iy+1,iz+1] + (_dz*(τzz[ix,iy,iz+1] - τzz[ix,iy,iz]) + _dx*(τxz[ix+1,iy,iz] - τxz[ix,iy,iz]) + _dy*(τyz[ix,iy+1,iz] - τyz[ix,iy,iz]) - _dz*(Pt[ix+1,iy+1,iz+1] - Pt[ix+1,iy+1,iz]))*@av_zi_dt_Rho(ix,iy,iz)  end
    return
end

@parallel_indices (ix,iy,iz) function compute_Res!(∇V::Data.Array, Rx::Data.Array, Ry::Data.Array, Rz::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, τxz::Data.Array, τyz::Data.Array, _dx::Data.Number, _dy::Data.Number, _dz::Data.Number)
    if (ix<=size(∇V,1) && iy<=size(∇V,2) && iz<=size(∇V,3))  ∇V[ix,iy,iz] = _dx*(Vx[ix+1,iy,iz] - Vx[ix,iy,iz]) + _dy*(Vy[ix,iy+1,iz] - Vy[ix,iy,iz]) + _dz*(Vz[ix,iy,iz+1] - Vz[ix,iy,iz]) end
    if (ix<=size(Rx,1) && iy<=size(Rx,2) && iz<=size(Rx,3))  Rx[ix,iy,iz] = _dx*(τxx[ix+1,iy,iz] - τxx[ix,iy,iz]) + _dy*(τxy[ix,iy+1,iz] - τxy[ix,iy,iz]) + _dz*(τxz[ix,iy,iz+1] - τxz[ix,iy,iz]) - _dx*(Pt[ix+1,iy+1,iz+1] - Pt[ix,iy+1,iz+1]) end
    if (ix<=size(Ry,1) && iy<=size(Ry,2) && iz<=size(Ry,3))  Ry[ix,iy,iz] = _dy*(τyy[ix,iy+1,iz] - τyy[ix,iy,iz]) + _dx*(τxy[ix+1,iy,iz] - τxy[ix,iy,iz]) + _dz*(τyz[ix,iy,iz+1] - τyz[ix,iy,iz]) - _dy*(Pt[ix+1,iy+1,iz+1] - Pt[ix+1,iy,iz+1]) end
    if (ix<=size(Rz,1) && iy<=size(Rz,2) && iz<=size(Rz,3))  Rz[ix,iy,iz] = _dz*(τzz[ix,iy,iz+1] - τzz[ix,iy,iz]) + _dx*(τxz[ix+1,iy,iz] - τxz[ix,iy,iz]) + _dy*(τyz[ix,iy+1,iz] - τyz[ix,iy,iz]) - _dz*(Pt[ix+1,iy+1,iz+1] - Pt[ix+1,iy+1,iz]) end
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

@views function Stokes3D()
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0  # domain extends
    ξ          = 1.0               # Maxwell relaxation time
    μs0        = 1.0               # matrix viscosity
    μsi        = 1e-3              # inclusion viscosity
    G          = 1.0               # elastic shear modulus
    εbg        = 1.0               # background strain-rate
    dt         = μs0/(G*ξ)
    # Numerics
    nt         = 1#5                 # number of time steps
    iterMax    = 200#2e5               # maximum number of pseudo-transient iterations
    nout       = 2000               # error checking frequency
    Re         = 6.0π              # Reynolds number
    r          = 1.0               # Bulk to shear elastic modulus ratio
    CFL        = 0.8/sqrt(3)       # CFL number
    ε          = 1e-8              # nonlinear absolute tolerence
    # nx, ny, nz = 127, 127, 127     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    # Derived numerics
    dx, dy, dz = lx/nx, ly/ny, lz/nz      # cell sizes
    max_lxyz   = max(lx,ly,lz)
    Vpdt       = min(dx,dy,dz)*CFL
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz
    # Array allocations
    Pt         = @zeros(nx  ,ny  ,nz  )
    dt_Rho     = @zeros(nx  ,ny  ,nz  )
    Gdt        = @zeros(nx  ,ny  ,nz  )
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
    Rx         = @zeros(nx-1,ny-2,nz-2)
    Ry         = @zeros(nx-2,ny-1,nz-2)
    Rz         = @zeros(nx-2,ny-2,nz-1)
    Mus2       = @zeros(nx  ,ny  ,nz  )
    Musτ       = @zeros(nx  ,ny  ,nz  )
    # Initial conditions
    Rad2       =  zeros(nx  ,ny  ,nz  )
    Vx         =  zeros(nx+1,ny  ,nz  )
    Vz         =  zeros(nx  ,ny  ,nz+1)
    Rad2      .= [(((ix-1)*dx +0.5*dx -0.5*lx)^2 + ((iy-1)*dy +0.5*dy -0.5*ly)^2 + ((iz-1)*dz +0.5*dz -0.5*lz)^2) for ix=1:size(Rad2,1), iy=1:size(Rad2,2), iz=1:size(Rad2,3)]
    Vx         = Data.Array( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2), iz=1:size(Vx,3)] )
    Vz         = Data.Array(  εbg.*[((iz-1)*dz -0.5*lz) for ix=1:size(Vz,1), iy=1:size(Vz,2), iz=1:size(Vz,3)] )
    Mus        = μs0*ones(nx,ny,nz)    
    Mus[Rad2.<1.0] .= μsi
    Mus        = Data.Array(Mus)
    Mus2      .= Mus
    for ism=1:10#15
        @parallel smooth!(Mus2, Mus, 1.0)
        Mus, Mus2 = Mus2, Mus
    end
    Musτ      .= Mus
    @parallel compute_maxloc!(Musτ, Mus)
    @parallel (1:size(Musτ,2), 1:size(Musτ,3)) bc_x!(Musτ)
    @parallel (1:size(Musτ,1), 1:size(Musτ,3)) bc_y!(Musτ)
    @parallel (1:size(Musτ,1), 1:size(Musτ,2)) bc_z!(Musτ)
    nx_1, nx_2, ny_1, ny_2, nz_1, nz_2 = nx-1, nx-2, ny-1, ny-2, nz-1, nz-2
    # Preparation of visualisation
    if do_viz || do_save_viz
        ENV["GKSwstype"]="nul"; if do_viz !ispath("../../figures") && mkdir("../../figures") end
        y_sl2, y_sl = Int(ceil((ny-2)/2)), Int(ceil(ny/2))
        xc, zc  = dx/2:dx:(lx-dx/2), dz/2:dz:(lz-dz/2)
        xv, zv  = 0:dx:lx, 0:dz:lz
    end
    # Time loop
    @parallel compute_iter_params!(dt_Rho, Gdt, Musτ, Vpdt, G, dt, Re, r, max_lxyz)
    t=0.0; ittot=0; evo_t=[]; evo_τzz=[]; t_tic = 0.0
    for it = 1:nt
        err=2*ε; iter=0; err_evo1=[]; err_evo2=[]
        @parallel assign_τ!(τxx, τyy, τzz, τxy, τxz, τyz, τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o)
        # Pseudo-transient iteration
        while err > ε && iter <= iterMax
            if (it==1 && iter==11) t_tic = Base.time() end
            @parallel compute_Pt_τ!(Pt, τxx, τyy, τzz, τxy, τxz, τyz, τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, Vx, Vy, Vz, Mus, Gdt, r, G, dt, _dx, _dy, _dz)
            @parallel compute_V!(Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, dt_Rho, _dx, _dy, _dz, nx_1, nx_2, ny_1, ny_2, nz_1, nz_2)
            @parallel (1:size(Vy,2), 1:size(Vy,3)) bc_x!(Vy)
            @parallel (1:size(Vz,2), 1:size(Vz,3)) bc_x!(Vz)
            @parallel (1:size(Vx,1), 1:size(Vx,3)) bc_y!(Vx)
            @parallel (1:size(Vz,1), 1:size(Vz,3)) bc_y!(Vz)
            @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_z!(Vx)
            @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_z!(Vy)
            iter += 1
            if iter % nout == 0
                @parallel compute_Res!(∇V, Rx, Ry, Rz, Vx, Vy, Vz, Pt, τxx, τyy, τzz, τxy, τxz, τyz, _dx, _dy, _dz)
                norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_Rz = norm(Rz)/length(Rz); norm_∇V = norm(∇V)/length(∇V)
                err = maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])
                if isnan(err) error("NaN") end
                push!(err_evo1,maximum([norm_Rx, norm_Ry, norm_Rz, norm_∇V])); push!(err_evo2,iter)
                @printf("Step = %d, iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n", it, iter, err, norm_Rx, norm_Ry, norm_Rz, norm_∇V)
            end
        end
        ittot += iter; t += dt
        # push!(evo_t, t); push!(evo_τzz, maximum(τzz))
    end
    # Performance
    wtime    = Base.time() - t_tic
    A_eff    = (10*2 + 6*1 + 3)/1e9*nx*ny*nz*sizeof(Data.Number) # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime/(ittot-10)                           # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                             # Effective memory throughput [GB/s]
    @printf("Total iters = %d (%d steps), time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", ittot, nt, wtime, round(T_eff, sigdigits=3))
    # Visualisation
    if do_viz
        p1 = heatmap(xc, zc, Array(Pt)[:,y_sl,:]', aspect_ratio=1, xlims=(xc[1],xc[end]), zlims=(zc[1],zc[end]), c=:viridis, title="Pressure")
        p2 = heatmap(xc, zv, Array(Vz)[:,y_sl,:]', aspect_ratio=1, xlims=(xc[1],xc[end]), zlims=(zc[1],zc[end]), c=:viridis, title="Vz")
        p4 = heatmap(xc[2:end-1], xv[2:end-1], log10.(abs.(Array(Rz)[:,y_sl2,:]')), aspect_ratio=1,  xlims=(xc[2],xc[end-1]), zlims=(zc[2],zc[end-1]), c=:viridis, title="log10(Rz)")
        #p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        p3 = plot(evo_t, evo_τzz, legend=false, xlabel="time", ylabel="max(τzz)", linewidth=0, markershape=:circle, framestyle=:box, markersize=3)
            #plot!(evo_t, 2.0.*εbg.*μs0.*(1.0.-exp.(.-evo_t.*G./μs0)), linewidth=2.0) # analytical solution
        plot(p1, p2, p4, p3)
        savefig("../../figures/Stokes_3D_ve_perf_$(nx).png")
    end
    if do_save
        !ispath("../../output") && mkdir("../../output")
        open("../../output/out_Stokes3D_ve_perf.txt","a") do io
            println(io, "1 $(nx) $(ny) $(nz) $(ittot) $(wtime) $(A_eff) $(wtime_it) $(T_eff)")
        end
    end
    if do_save_viz
        !ispath("../../out_visu") && mkdir("../../out_visu")
        matwrite("../../out_visu/Stokes_3D_ve_perf.mat", Dict("Pt_3D"=> Array(Pt), "Mus_3D"=> Array(Mus), "Txz_3D"=> Array(τxz), "Vz_3D"=> Array(Vz), "dx_3D"=> dx, "dy_3D"=> dy, "dz_3D"=> dz); compress = true)
    end
    return
end

Stokes3D()
