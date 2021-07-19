clear
%% symbolic functions and variables
syms p(t,x) v_x(t,x) tau_xx(t,x)
syms K G rho eta k Lx Re r v0 positive
syms lambda_k
%% governing equations
eq1      = 1/K*diff(p(t,x)     ,t) + diff(          v_x(t,x),x);           % mass balance
eq2      = rho*diff(v_x(t,x)   ,t) + diff(p(t,x)-tau_xx(t,x),x);           % momentum balace
eq3      = 1/G*diff(tau_xx(t,x),t) + 1/eta*tau_xx(t,x) - 2*diff(v_x(t,x),x); % Maxwell rheological model
%% equation for velocity
tmp      = diff(eq2,t)/G + diff(eq3,x) + eq2/eta - diff(eq1,x)*K/G;
eq_v     = expand(diff(tmp,t)*eta - diff(eq1,x)*K);
%% scales and nondimensional variables
K        = r*G;                                                            % bulk modulus from modulus ratio r
V_sh     = sqrt((K+2*G)/rho);                                              % velocity scale - shear wave velocity
rho      = solve(Re == rho*V_sh*Lx/eta,rho);                               % density from Reynolds number Re
%% dispersion relation
v_x(t,x) = v0*exp(-lambda_k*V_sh*t/Lx)*sin(pi*k*x/Lx);                     % Fourier term
disp_rel = subs(subs(eq_v/v_x(t,x)));
cfs      = coeffs(disp_rel,lambda_k);
cfs      = fliplr(simplify(cfs/cfs(end)));
disp_rel = poly2sym(cfs,lambda_k);
%% optimal iteration parameters
a        = cfs(4);
b        = cfs(3);
c        = cfs(2);
d        = cfs(1);
discrim  = b^2*c^2 - 4*a*c^3 - 4*b^3*d - 27*a^2*d^2 + 18*a*b*c*d;          % cubic polynomial discriminant
cfs2     = coeffs(discrim,Re);
a2       = cfs2(3);
b2       = cfs2(2);
c2       = cfs2(1);
discrim2 = b2^2 - 4*a2*c2;                                                 % quadratic polynomial discriminant
r_opt    = solve(discrim2,r,'PrincipalValue',true);
Re_opt   = simplify(solve(subs(discrim,{r,k},{r_opt,1}),Re,'PrincipalValue',true));
%% evaluate the solution numerically
num_cfs  = matlabFunction(subs(cfs,k,1));
[Re2,r2] = ndgrid(linspace(double(Re_opt)/2,  double(Re_opt)*3/2,1001)...
    ,             linspace(0               ,2*double(r_opt)     ,1001));   % create uniform grid of Re and r values
num_lam  = arrayfun(@(Re,r)(min(real(roots(num_cfs(Re,r))))),Re2,r2);      % compute minimum of real part of roots
%% plot the spectral abscissa
figure(1);clf;colormap cool
contourf(Re2,r2,num_lam,15,'LineWidth',1);axis square
cb = colorbar;
cb.Label.Interpreter='latex';
cb.Label.String='$\mathrm{max}\{\Re(\lambda_k)\}$';
cb.Label.FontSize=16;
xline(double(Re_opt),'w--','LineWidth',2,'Alpha',1)
yline(double(r_opt),'w--','LineWidth',2,'Alpha',1)
hold on
plot(double(Re_opt),double(r_opt),'wo','MarkerSize',7,'MarkerFaceColor','w')
hold off
ax = gca;
xlabel('$Re$','Interpreter','latex');ylabel('$r$','Interpreter','latex')
xticks([double(Re_opt)/2 double(Re_opt) double(Re_opt)*3/2])
yticks([0 double(r_opt) 2*double(r_opt)])
xticklabels({'$9\sqrt{3}\pi/8$','$9\sqrt{3}\pi/4$','$9\sqrt{3}\pi/2$'})
yticklabels({'0','1/4','1/2'})
ax.XAxis.TickLabelInterpreter = 'latex';
ax.YAxis.TickLabelInterpreter = 'latex';
set(gca,'FontSize',14)
title(['$' latex(disp_rel) '$'],'Interpreter','latex','FontSize',16)