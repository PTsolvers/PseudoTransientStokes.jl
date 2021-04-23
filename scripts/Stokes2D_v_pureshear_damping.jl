# Initialisation
using Plots, Printf, Statistics, LinearAlgebra
Dat     = Float64  # Precision (double=Float64 or single=Float32)
damping = 1        # 1: damping applied on momentum equation, 2: damping applied on fluxes
# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
# 2D Stokes routine
@views function Stokes2D_v()
    # Physics
    Lx, Ly  = 1.0, 1.0  # domain size
    radi    = 0.01      # inclusion radius
    μ0      = 1.0       # viscous viscosity
    μi      = μ0/10.0   # elastic shear modulus perturbation
    εbg     = 1.0       # background strain-rate
    # Numerics
    nt      = 1         # number of time steps
    nx, ny  = 127, 127    # numerical grid resolution
    Vdmp    = 4.0       # convergence acceleration (damping)
    Ptsc    = 8.0       # iterative time step limiter
    tolnl   = 1e-10      # nonlinear tolerence
    iterMax = 1e5       # max number of iters
    nout    = 200       # check frequency
    # Preprocessing
    dx, dy  = Lx/nx, Ly/ny
    # Array initialisation
    Pt      = zeros(Dat, nx  ,ny  )
    ∇V      = zeros(Dat, nx  ,ny  )
    Vx      = zeros(Dat, nx+1,ny  )
    Vy      = zeros(Dat, nx  ,ny+1)
    Exx     = zeros(Dat, nx  ,ny  )
    Eyy     = zeros(Dat, nx  ,ny  )
    Exy_in  = zeros(Dat, nx-1,ny-1)
    Txx     = zeros(Dat, nx  ,ny  )
    Tyy     = zeros(Dat, nx  ,ny  )
    Txy     = zeros(Dat, nx+1,ny+1)
    Tii     = zeros(Dat, nx  ,ny  )
    Rx      = zeros(Dat, nx-1,ny  )
    Ry      = zeros(Dat, nx  ,ny-1)
    dVxdt   = zeros(Dat, nx-1,ny  )
    dVydt   = zeros(Dat, nx  ,ny-1)
    dtVx    = zeros(Dat, nx-1,ny  )
    dtVy    = zeros(Dat, nx  ,ny-1)
    dtPt    = zeros(Dat, nx  ,ny  )
    Rog     = zeros(Dat, nx  ,ny  )
    Mus     = μ0*ones(Dat, nx, ny )
    # new
    Kdt     = zeros(Dat, nx  ,ny  )
    Gdt     = zeros(Dat, nx  ,ny  )
    dVx     = zeros(Dat, nx-1,ny  )
    dVy     = zeros(Dat, nx  ,ny-1)
    dt_rho  = zeros(Dat, nx  ,ny  )
    dt_rhox = zeros(Dat, nx-1,ny  )
    dt_rhoy = zeros(Dat, nx  ,ny-1)
    # Initialisation
    xc, yc  = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    xc, yc  = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    xv, yv  = LinRange(0.0, Lx, nx+1), LinRange(0.0, Ly, ny+1)
    (Xvx,Yvx) = ([x for x=xv,y=yc], [y for x=xv,y=yc])
    (Xvy,Yvy) = ([x for x=xc,y=yv], [y for x=xc,y=yv])
    rad       = (xc.-Lx./2).^2 .+ (yc'.-Ly./2).^2
    Mus[rad.<radi].= μi
    Vx     .=   εbg.*Xvx
    Vy     .= .-εbg.*Yvy
    # Ludo's trick
    nloc    = 1
    Musm    = zeros(size(Mus))
    Mus0    = zeros(size(Mus))
    Musm   .= Mus
    Mus0   .= Mus
    for iloc=1:nloc
        Musm[2:end-1,2:end-1] = max.( max.( max.(Mus0[1:end-2,2:end-1] , Mus0[3:end  ,2:end-1]) , Mus0[2:end-1,2:end-1] ) , max.(Mus0[2:end-1,1:end-2] , Mus[2:end-1,3:end  ]) );
        Musm[1,:] = Musm[2,:]; Musm[end,:] = Musm[end-1,:]
        Musm[:,1] = Musm[:,2]; Musm[:,end] = Musm[:,end-1]
        Mus0 .= Musm
    end
    if damping==1
        dtVx   .= min(dx,dy)^2.0./av_xa(Musm)./4.1./2.0
        dtVy   .= min(dx,dy)^2.0./av_ya(Musm)./4.1./2.0
        dtPt   .= 4.1*Musm/max(nx,ny)/Ptsc
    end
    if damping==2
        sc1      = 1.0 
        cfl      = 1.0 /(2.0 + 3.0 *log10(μ0/μi))
        Kdt     .= sc1.*2*pi.*dx*cfl./Lx.*Musm
        Gdt     .=      4*pi.*dx*cfl./Lx.*Musm
        dt_rho  .= (dx*cfl).^2 ./ ( Kdt .+ Gdt./(1.0 .+ Gdt./Musm ) )
        dt_rhox .= av_xa(dt_rho)
        dt_rhoy .= av_ya(dt_rho)
    end
    # Time loop
    for it = 1:nt
        iter=1; err=2*tolnl; err_evo1=[]
        while (err>tolnl && iter<=iterMax)
            # divergence - pressure
            ∇V    .= diff(Vx, dims=1)./dx .+ diff(Vy, dims=2)./dy
            if damping==1
                Pt    .= Pt .- dtPt.*∇V
            end
            if damping==2
                Pt    .= Pt .-  Kdt.*∇V
            end
            # strain rates
            Exx    .= diff(Vx, dims=1)./dx .- 1.0/3.0*∇V
            Eyy    .= diff(Vy, dims=2)./dy .- 1.0/3.0*∇V
            Exy_in .= 0.5.*(diff(Vx[2:end-1,:], dims=2)./dy .+ diff(Vy[:,2:end-1], dims=1)./dx)
            # stresses
            if damping==1  
                Txx   .= 2.0.*Mus.*Exx
                Tyy   .= 2.0.*Mus.*Eyy
                Txy[2:end-1,2:end-1] .= 2.0.*av(Mus).*Exy_in
            end
            if damping==2
                Txx                  .= (Txx .+ 2.0.*Gdt.*Exx) ./ (1.0 .+ Gdt./Mus)
                Tyy                  .= (Tyy .+ 2.0.*Gdt.*Eyy) ./ (1.0 .+ Gdt./Mus)
                Txy[2:end-1,2:end-1] .= (Txy[2:end-1,2:end-1] .+ 2.0.*av(Gdt).*Exy_in) ./ (1.0 .+ av(Gdt./Mus))
            end
            Tii   .= sqrt.(0.5*(Txx.^2 .+ Tyy.^2) .+ av(Txy).^2)
            # velocities
            Rx    .= .-diff(Pt, dims=1)./dx .+ diff(Txx, dims=1)./dx .+ diff(Txy[2:end-1,:], dims=2)./dy
            Ry    .= .-diff(Pt, dims=2)./dy .+ diff(Tyy, dims=2)./dy .+ diff(Txy[:,2:end-1], dims=1)./dx .+ av_ya(Rog)
            if damping==1
                dVxdt .= dVxdt.*(1-Vdmp/nx) .+ Rx
                dVydt .= dVydt.*(1-Vdmp/ny) .+ Ry
                dVx   .= dtVx.*dVxdt
                dVy   .= dtVy.*dVydt
            end
            if damping==2
                dVx   .= dt_rhox .* Rx
                dVy   .= dt_rhoy .* Ry
            end
            Vx[2:end-1,:] .= Vx[2:end-1,:] .+ dVx
            Vy[:,2:end-1] .= Vy[:,2:end-1] .+ dVy
            # convergence check
            if mod(iter, nout)==0
                global max_Rx, max_Ry, max_divV
                norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
                err = maximum([norm_Rx, norm_Ry, norm_∇V])
                push!(err_evo1, err)
                @printf("iter/nx = %d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, ∇V=%1.3e] \n", iter/nx, err, norm_Rx, norm_Ry, norm_∇V)
            end
            iter+=1;
        end
        # Plotting
        p1 = heatmap(xv, yc, Vx'  , aspect_ratio=1, xlims=(0, Lx), ylims=(dy/2, Ly-dy/2), c=:inferno, title="Vx")
        p2 = heatmap(xv, yv, Txy' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="τxy")
        p3 = heatmap(xc, yc, Mus' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="μ")
        p4 = heatmap(xc, yc, Tii' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="τii")
        display(plot(p1, p2, p3, p4, background_color=:transparent, foreground_color=:gray))
    end
end

@time Stokes2D_v()
