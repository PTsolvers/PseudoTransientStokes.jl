clear;figure(1);clf;load('roma.mat')
%setup
set(gcf,'Units','centimeters','Position',[12 2 12 18],'PaperUnits','centimeters','PaperPosition',[0 0 12 18]);
fs = 9;
% physics and numerics
lx = 10; ly = 10; lz = 10; nx = 2048;
% vis
subs   = 1:3:7;
vrs    = 1:4:9;
rems   = 0.5:0.25:1.5;
nincls = [1 14 46];
data   = readmatrix('../output/out_Stokes2D_strong_param.txt');
sz     = [numel(vrs),numel(subs),numel(rems)];
Vr     = reshape(data(:,1),sz);
Nsub   = reshape(data(:,2),sz);
Re     = reshape(data(:,3),sz);
Iters  = reshape(data(:,4),sz);
tiledlayout(numel(subs),2,'TileSpacing','compact','Padding','normal')
for iSub = 1:numel(subs)
    % results
    nexttile
    load(sprintf('../output/VISCR_9/NSUB_%d/REMULT_1.0/Stokes2D_strong.mat',subs(iSub)))
    dx = dx_2D; xc = dx/2:dx:lx-dx/2; xv = dx:dx:lx-dx;
    dy = dy_2D; yc = dy/2:dy:ly-dy/2; yv = dy:dy:ly-dy;
    [Xc,Yc] = meshgrid(xc,yc);
    phi = nincls(iSub)*0.005;
    rhog_eff = phi*0.5 + (1 - phi)*1;
    dP = Pt_2D + rhog_eff*(Yc' - ly/2);
    imagesc(xv,yv,dP');shading flat;axis image;caxis([-0.2 0.2])
    hold on;
    Vxc = 0.5*(Vx_2D(1:end-1,:)+Vx_2D(2:end,:));
    Vyc = 0.5*(Vy_2D(:,1:end-1)+Vy_2D(:,2:end));
    l=streamslice(Xc,Yc,Vxc',Vyc',1);
    set(l,'Color','k','LineWidth',0.5);
    hold off;
    xlim([0 lx]);ylim([0 ly]);
    yticks([0 ly]);ylabel('\bfly')
    if iSub == 1; xticks([0 lx]);xlabel('\bflx');else; xticks([]); end
    if iSub == numel(subs);cb1=colorbar('Location','southoutside'); cb1.Label.String = '\bfp';cb1.Label.FontSize=fs;end
    text(-0.3,1.05,['\bf(' char('a'+iSub-1) ')'],'Units','normalized','FontName','Courier')
    set(gca,'FontName','Courier','FontSize',fs,'YDir','normal','XAxisLocation','top')%,'ColorScale','log')
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
cb2.Position(2)=cb1.Position(2);
% exportgraphics(gcf,'fig_stokes_systematics_shear.png','Resolution',300)