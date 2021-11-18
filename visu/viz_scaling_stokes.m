clear

% load scaling_data
fid = fopen('../output/out_Stokes2D_ve.txt','r');
Stokes_2D = fscanf(fid, '%d %d %d %d', [4 Inf]);
fclose(fid);
fid = fopen('../output/out_Stokes3D_ve.txt','r');
Stokes_3D = fscanf(fid, '%d %d %d %d %d', [5 Inf]);
fclose(fid);

fid = fopen('../output/out_Stokes2D_ve0.txt','r');
Stokes_2Do = fscanf(fid, '%d %d %d %d', [4 Inf]);
fclose(fid);
fid = fopen('../output/out_Stokes3D_ve0.txt','r');
Stokes_3Do = fscanf(fid, '%d %d %d %d %d', [5 Inf]);
fclose(fid);


FS = 20;

fig1 = 1;
fig2 = 0;

%%
if fig1==1
figure(2),clf,set(gcf,'color','white','pos',[1400 10 500 400])

semilogx(Stokes_2D(1,2:end), Stokes_2D(3,2:end)./Stokes_2D(1,2:end)./Stokes_2D(4,2:end),'-o', ...
         Stokes_3D(1,1:end), Stokes_3D(4,1:end)./Stokes_3D(1,1:end)./Stokes_3D(5,1:end),'-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
ylabel({' ';'\bf{iter_{tot}/nt/nx}'}, 'fontsize',FS)
ylim([15 20]) % ylim([3 15])
lg=legend('2D', '3D'); set(lg,'box','off')
set(gca, 'XTick',Stokes_2D(1,2:end))
xtickangle(45)
xlabel('\bf{nx}', 'fontsize',FS)
title('visco-elastic Stokes')
set(gca,'fontname','Courier')

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_stokes_ve_scale23D','-dpng','-r300')
end

if fig2==1
figure(2),clf,set(gcf,'color','white','pos',[1400 10 900 450])

sp1=subplot(121);
semilogx(Stokes_2Do(1,2:end), Stokes_2Do(3,2:end)./Stokes_2Do(1,2:end)./Stokes_2Do(4,2:end),'-o', ...
         Stokes_2D(1,2:end), Stokes_2D(3,2:end)./Stokes_2D(1,2:end)./Stokes_2D(4,2:end),'-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
ylabel({' ';'\bf{iter_{tot}/nt/nx}'}, 'fontsize',FS)
ylim([2 20])
lg=legend('accel. 1', 'accel. 2'); set(lg,'box','off')
set(gca, 'XTick',Stokes_2Do(1,2:end))
xtickangle(45)
xlabel('\bf{nx}', 'fontsize',FS)
title('visco-elastic Stokes 2D')
set(gca,'fontname','Courier')

sp2=subplot(122);
semilogx(Stokes_3Do(1,1:end), Stokes_3Do(4,1:end)./Stokes_3Do(1,1:end)./Stokes_3Do(5,1:end),'-o', ...
         Stokes_3D(1,1:end), Stokes_3D(4,1:end)./Stokes_3D(1,1:end)./Stokes_3D(5,1:end),'-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
% ylabel({' ';'\bf{iter_{tot}/nt/nx}'}, 'fontsize',FS)
ylim([2 20])
lg=legend('accel. 1', 'accel. 2'); set(lg,'box','off')
set(gca, 'XTick',Stokes_3Do(1,1:end),'Yticklabel',[])
xtickangle(45)
xlabel('\bf{nx}', 'fontsize',FS)
title('visco-elastic Stokes 3D')
set(gca,'fontname','Courier')

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_stokes_ve_scale','-dpng','-r300')
end