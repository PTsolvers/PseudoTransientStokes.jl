clear;figure(1);clf;colormap jet
subs   = 1:5;
nincls = [1 3 8 14 23];
vrs    = 9:2:9;
rems   = 0.5:0.1:1.5;
lx     = 10;
ly     = 10;
tiledlayout(4,numel(subs),'TileSpacing','compact','Padding','compact')
for isub = subs
    for ivr = vrs
        for irem = 1:numel(rems)
            Remlt = rems(irem);
            simdir = sprintf('../output/VISCR_%d/NSUB_%d/REMULT_%.1f',ivr,isub,Remlt);
            if irem == numel(rems)
                data_2D = load([simdir '/Stokes2D.mat']);
                Pt_2D  = data_2D.Pt_2D;
                Mu_2D  = data_2D.Mus_2D;
                dx = data_2D.dx_2D; xc_2D = -lx/2+dx/2:dx:lx/2-dx/2;
                dy = data_2D.dy_2D; yc_2D = -ly/2+dy/2:dy:ly/2-dy/2;
                [x2D,y2D] = ndgrid(xc_2D,yc_2D);
                nexttile;pcolor(x2D,y2D,Pt_2D);shading flat;axis image;colorbar
                caxis([-5 5])
                title(isub)
            end
            itdata = readmatrix([simdir '/out_Stokes2D_param.txt'],'FileType','text','Delimiter',' ');
            ittot(isub,irem)  = itdata(3)/2048;
            errtot(isub,irem) = itdata(4);
        end
    end
    disp(isub)
end
[subs2,rems2] = ndgrid(nincls*2.5,rems);
% nexttile([1 2]);
contourf(subs2,rems2,ittot,10:20,'LineWidth',1);shading interp;caxis([10 20]);cb=colorbar;
xlabel('Weak/Total area ratio, %')
ylabel('Re/Re_{opt}')
cb.Label.String = 'n_{iter}/n_x';cb.Label.FontSize = 14;
cb.FontSize = 14;
set(gca,'FontSize',14)
[m,I] = min(ittot,[],2);
hold on
plot(nincls*2.5,rems(I),'w-','LineWidth',2)
hold off
drawnow