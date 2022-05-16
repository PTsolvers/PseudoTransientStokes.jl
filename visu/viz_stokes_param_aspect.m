clear;load('roma.mat')
%setup
fs = 10;                           % font size
% figure parameters
nt       = 5;
sims     = {'aspect_ratio'};
fact     = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
subs     = [1 2 4 8];                       % number of grid subdivisions
resol    = [127, 255, 511, 1023, 2047];
% vis
for ifig = 1:numel(sims)
    figure(ifig);clf
    set(gcf,'Units','centimeters','Position',[(5+13*(ifig-1)) 2 8 18],'PaperUnits','centimeters','PaperPosition',[0 0 8 18]);
    data   = readmatrix(sprintf('../output/out_Stokes2D_ve_%s_param.txt',sims{ifig}));
%     data   = data(:,2:end);
    sz     = [numel(fact),numel(subs),numel(resol)];
    nx     = reshape(data(:,1),sz);
    ny     = reshape(data(:,2),sz);
    nsub   = reshape(data(:,3),sz);
    nfact  = reshape(data(:,4),sz);
    iters  = reshape(data(:,5),sz);
    error  = reshape(data(:,7),sz);
    tiledlayout(numel(resol),1,'TileSpacing','compact','Padding','normal')
    for iSub = 1:numel(resol)
        nexttile
        plot(min(iters(:,:,iSub))./nt./min(nx(:,:,iSub)),'-o','LineWidth',1.6,'MarkerSize',3)
        xlim([0.9 4.1]); ylim([6 20]);
        yticks([8 12 16]); yticklabels([8 12 16])
        if iSub == numel(resol);xticks(1:4);xticklabels(subs);xlabel('\bfaspect ratio (lx/ly)');else; xticklabels([]); end
        text(0.02,0.88,['\bf(' char('a'+iSub-1) ')'],'Units','normalized','FontName','Courier')
        text(0.63,0.88,['\bfny = ' num2str(resol(iSub))],'Units','normalized','FontName','Courier')
        set(gca,'FontName','Courier','FontSize',fs,'YDir','normal','XAxisLocation','bottom','LineWidth',0.8)
        if iSub == 3; ylabel('\bfiter_{tot}/nt/nx'); end
        drawnow
    end
    % save figure
    print('-dpng',sprintf('fig_stokes_systematics_%s.png',sims{ifig}),'-r300')
end