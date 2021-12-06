clear;load('roma.mat')
%setup
fs = 9;                           % font size
% physics and numerics
lx = 10; ly = 10; lz = 10; nx = 2048;
% figure parameters
sims     = {'shear_weak','gravity_weak'};
Pgrad    = [0 1];                 % remove pressure gradient for visualisation
Prange   = {[-2 2], [-0.2 0.2]};  % pressure color scale
subs     = 1:3:7;                 % number of grid subdivisions
vrs      = 1:4:9;                 % viscosity ratio exponent
rems     = 0.5:0.25:1.5;          % Reynolds number
nincls   = [1 14 46];             % number of inclusions
phi_incl = 0.005;                 % one inclusion volume fraction
% vis
for ifig = 1:numel(sims)
    figure(ifig);clf
    set(gcf,'Units','centimeters','Position',[(5+13*(ifig-1)) 2 12 18],'PaperUnits','centimeters','PaperPosition',[0 0 12 18]);
    data   = readmatrix(sprintf('../output/out_Stokes2D_%s_param.txt',sims{ifig}));
    sz     = [numel(vrs),numel(subs),numel(rems)];
    Vr     = reshape(data(:,1),sz);
    Nsub   = reshape(data(:,2),sz);
    Re     = reshape(data(:,3),sz);
    Iters  = reshape(data(:,4),sz);
    tiledlayout(numel(subs),2,'TileSpacing','compact','Padding','normal')
    for iSub = 1:numel(subs)
        % draw pressure and streamlines
        nexttile
        load(sprintf('../out_visu/%s/VISCR_9_NSUB_%d_REMULT_1.0/Stokes2D.mat',sims{ifig},subs(iSub)))
        dx = dx_2D; xc = dx/2:dx:lx-dx/2; xv = dx:dx:lx-dx;
        dy = dy_2D; yc = dy/2:dy:ly-dy/2; yv = dy:dy:ly-dy;
        [Xc,Yc] = meshgrid(xc,yc);
        phi = nincls(iSub)*phi_incl;
        rhog_eff = phi*0.5 + (1 - phi)*1;
        dP = Pt_2D + Pgrad(ifig)*rhog_eff*(Yc' - ly/2);
        imagesc(xv,yv,dP');shading flat;axis image;caxis(Prange{ifig})
        hold on;
        Vxc = 0.5*(Vx_2D(1:end-1,:)+Vx_2D(2:end,:));
        Vyc = 0.5*(Vy_2D(:,1:end-1)+Vy_2D(:,2:end));
        l=streamslice(Xc,Yc,Vxc',Vyc',1);
        set(l,'Color','k','LineWidth',0.5);
        hold off;
        xlim([0 lx]);ylim([0 ly]);
        yticks([0 ly]);ylabel('\bfly')
        if iSub == 1          ;xticks([0 lx]);xlabel('\bflx');else; xticks([]); end
        if iSub == numel(subs);cb1=colorbar('Location','southoutside'); cb1.Label.String = '\bfp';cb1.Label.FontSize=fs;end
        text(-0.3,1.05,['\bf(' char('a'+iSub-1) ')'],'Units','normalized','FontName','Courier')
        set(gca,'FontName','Courier','FontSize',fs,'YDir','normal','XAxisLocation','top')
        colormap(gca,roma)
        % iter count
        nexttile
        imagesc(vrs,rems,squeeze(Iters(:,iSub,:))'/nx);shading flat
        hold on
        [~,I] = min(squeeze(Iters(:,iSub,:)),[],2);
        Resc = squeeze(Re(:,iSub,:));
        plot(vrs,rems(I),'w-o','LineWidth',1,'MarkerSize',2);
        hold off
        axis square;caxis([10 50]);
        yticks(0.5:0.5:1.5);ylabel('\bfRe/Re_{opt}')
        if iSub == 1;xticks(vrs);xlabel('\bf\mu^{0}_s/\mu^{inc}_s');else;xticks([]); end
        if iSub == numel(subs);cb2=colorbar('Location','southoutside');cb2.Label.String='\bfiter_{tot}/nx';cb2.Label.FontSize=fs;end
        set(gca,'FontName','Courier','FontSize',fs,'YDir','normal','XAxisLocation','top','YAxisLocation','right')
        colormap(gca,flip(roma))
        drawnow
    end
    % fix colorbar position on the right panel
    cb2.Position(2)=cb1.Position(2);
    % save figure
    exportgraphics(gcf,sprintf('fig_stokes_systematics_%s.png',sims{ifig}),'Resolution',300)
end