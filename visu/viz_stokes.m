clear

fig1=0;
fig2=1;

lx = 10; ly = 10; lz = 10;
data_2D = load('../out_visu/Stokes_2D_ve3.mat');
Pt_2D  = data_2D.Pt_2D;
Mu_2D  = data_2D.Mus_2D;
Txy_2D = data_2D.Txy_2D;
Vy_2D = data_2D.Vy_2D;
dx = data_2D.dx_2D; xc_2D = dx/2:dx:lx-dx/2;
dy = data_2D.dy_2D; yc_2D = dy/2:dy:ly-dy/2; yv_2D = 0:dy:ly; yv_2D2 = yv_2D - ly/2;
Vy_BG = repmat(yv_2D2,size(Vy_2D,1),1);

data_3D = load('../out_visu/Stokes_3D_ve3.mat');
Pt_3D  = data_3D.Pt_3D;
Mu_3D  = data_3D.Mus_3D;
Txz_3D = data_3D.Txz_3D;
Vz_3D = data_3D.Vz_3D;
dx = data_3D.dx_3D; xc_3D = dx+dx/2:dx:lx-dx-dx/2;
dy = data_3D.dy_3D; yc_3D = dy+dy/2:dy:ly-dy-dy/2;
dz = data_3D.dz_3D; zc_3D = dz+dz/2:dz:lz-dz-dz/2; zc_3D2 = zc_3D-lz/2;
[x3d,y3d,z3d] = ndgrid(xc_3D,yc_3D,zc_3D2);

FS = 20;
if fig1==1
figure(1),clf,set(gcf,'color','white','pos',[1400 100 1000 400]) 
%%2D
sp1 = subplot(121); imagesc(xc_2D, yc_2D, log10(Mu_2D)'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
set(gca, 'XTick', [0.2 9.9], 'XTicklabel', [0 10], 'fontsize',FS)
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'TickLength',[0,0])
set(gca,'fontname','Courier')
ylabel({'\bf{ly}';' ';' '}, 'fontsize',FS)
xlabel({' ';' ';'\bf{lx}'}, 'fontsize',FS)
text(-5,13,'a)','fontsize',FS+2,'fontname','Courier')

%%3D
sp2 = subplot(122);
nx = size(Vz_3D,1); ny = size(Vz_3D,2); nz = size(Vz_3D,3);
D = log10(Mu_3D);

s1 = fix( nx/2.0);
s2 = fix( ny/2.0);
s3 = fix( nz/3  );
s4 = fix( nz/2  );
s5 = fix( nz    );

hold on
slice(D( :  , :  ,1:s5),[],s2,s3),shading flat
slice(D( :  , :  ,1:s3),ny,[],[]),shading flat
slice(D(1:s2, :  ,1:s5),ny,[],[]),shading flat
slice(D( :  , :  ,1:s3),[],nx,[]),shading flat
slice(D( :  ,1:s1,1:s4),[],nx,[]),shading flat
slice(D(1:s2, :  , :  ),[],[],s5),shading flat
slice(D( :  ,1:s1, :  ),[],[],s4),shading flat
slice(D( :  , :  ,1:s4),s1,[],[]),shading flat
%%%
is1 = isosurface(D, -1);
his1 = patch(is1); set(his1,'CData',-3,'Facecolor','Flat','Edgecolor','none')
hold off
% caxis([-0.62 0.62])
set(gca, 'linewidth',1.4)
set(gca, 'Ticklength', [0 0])
set(gca, 'XTick', [10 nx-5], 'XTicklabel', [10 0], 'fontsize',FS)
set(gca, 'YTick', [10 ny-5], 'YTicklabel', [10 0], 'fontsize',FS)
set(gca, 'ZTick', [10 nz-5], 'ZTicklabel', [0 10], 'fontsize',FS)

% xlabel('Lz = 3000','fontsize',FS,'interpreter','latex')

% title('a) \quad\quad [x, y, z] correlation length = [5, 5, 5]','fontsize',FS+2,'interpreter','latex')
% title('b) \quad\quad [x, y, z] correlation length = [3, 3, 20]','fontsize',FS+2,'interpreter','latex')
% title('c) \quad\quad [x, y, z] correlation length = [20, 20, 3]','fontsize',FS+2,'interpreter','latex')

text(150,315,0,'\bf{ly}','fontsize',FS,'fontname','Courier') %xlabel('lx','fontsize',FS)
text(350,160,0,'\bf{lx}','fontsize',FS,'fontname','Courier') %ylabel('ly','fontsize',FS)
zlabel('\bf{lz}','fontsize',FS)
% title({'Stokes 3D',' '})

cb = colorbar;
% cb.Location = 'southoutside';
% cb.Limits = [-0.62 0.62];
% cb.Ticks = [0.05 0.95];
% cb.TickLabels = {'0' '1'};
posCB = get(cb,'position');
cb.Position = [posCB(1)*1.06 posCB(2)*2.5 posCB(3)*1.1 posCB(4)*0.4];
% set(cb,'fontsize',FS)
set(gca,'fontsize',FS,'fontname','Courier')
text(390,0,365,'b)','fontsize',FS+2,'fontname','Courier')
text(-400,0,80,'\bf{log_{10}\mu_{shear}}','fontsize',FS-2,'fontname','Courier')

box on
axis image
view(136,18)
camlight
% light
camproj perspective
light('position',[0.6 -1 1]);
% light('position',[1 -1 1]);
light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);

pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*0.9  pos1(2)*2.5 pos1(3)*0.6 pos1(4)*0.6])
pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)*1.0  pos2(2)*1.8 pos2(3)*0.8 pos2(4)*0.8])

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_stokes_ini','-dpng','-r300')
end

if fig2==1
figure(2),clf,set(gcf,'color','white','pos',[1400 100 900 900])
%% 2D
sp1 = subplot(321); imagesc(xc_2D, yv_2D, Vy_2D'-Vy_BG'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
colorbar
set(gca, 'XTick', [])
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'TickLength',[0,0])
set(gca,'fontname','Courier')
ylabel({'\bf{∆V_{vertical}}',' ','ly'}, 'fontsize',FS)
% title({'Stokes 2D',' '})
text(0.4,9.3,'a)','fontsize',FS+2,'fontname','Courier','color','w')

sp2 = subplot(323); imagesc(xc_2D, yc_2D, Pt_2D'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
colorbar
set(gca, 'XTick', [])
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'fontname','Courier')
set(gca,'TickLength',[0,0])
ylabel({'\bf{Pressure}',' ','ly'}, 'fontsize',FS)
text(0.4,9.3,'c)','fontsize',FS+2,'fontname','Courier','color','w')

sp3 = subplot(325); imagesc(xc_2D(2:end), yc_2D(2:end), Txy_2D'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
cb=colorbar;
posCB = get(cb,'position');
cb.Position = [posCB(1)*1.036 posCB(2)*0.82 posCB(3) posCB(4)*1.1];
set(gca, 'XTick', [0.2 9.9], 'XTicklabel', [0 10], 'fontsize',FS)
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'fontname','Courier')
set(gca,'TickLength',[0,0])
xlabel('\bf{lx}', 'fontsize',FS)
ylabel({'\bf{\tau_{shear}}',' ','ly'}, 'fontsize',FS)
text(0.4,9.3,'e)','fontsize',FS+2,'fontname','Courier','color','w')

%% 3D
sp4 = subplot(322);
nx = size(Vz_3D,1); ny = size(Vz_3D,2); nz = size(Vz_3D,3);
D = Vz_3D-z3d;

s1 = fix( nx/2.0);
s2 = fix( ny/2.0);
s3 = fix( nz/3  );
s4 = fix( nz/2  );
s5 = fix( nz    );

hold on
slice(D( :  , :  ,1:s5),[],s2,s3),shading flat
slice(D( :  , :  ,1:s3),ny,[],[]),shading flat
slice(D(1:s2, :  ,1:s5),ny,[],[]),shading flat
slice(D( :  , :  ,1:s3),[],nx,[]),shading flat
slice(D( :  ,1:s1,1:s4),[],nx,[]),shading flat
slice(D(1:s2, :  , :  ),[],[],s5),shading flat
slice(D( :  ,1:s1, :  ),[],[],s4),shading flat
slice(D( :  , :  ,1:s4),s1,[],[]),shading flat
%%%
is1 = isosurface(D, 0.45);
is2 = isosurface(D,-0.45);
his1 = patch(is1); set(his1,'CData',0.5,'Facecolor','Flat','Edgecolor','none')
his2 = patch(is2); set(his2,'CData',-0.5,'Facecolor','Flat','Edgecolor','none')
hold off
caxis([-0.62 0.62])
set(gca, 'linewidth',1.4)
set(gca, 'Ticklength', [0 0])
set(gca, 'XTick', [])%[10 nx-5], 'XTicklabel', [10 0], 'fontsize',FS)
set(gca, 'YTick', [])%[10 ny-5], 'YTicklabel', [10 0], 'fontsize',FS)
set(gca, 'ZTick', [10 nz-5], 'ZTicklabel', [0 10], 'fontsize',FS)

% xlabel('Lz = 3000','fontsize',FS,'interpreter','latex')

% title('a) \quad\quad [x, y, z] correlation length = [5, 5, 5]','fontsize',FS+2,'interpreter','latex')
% title('b) \quad\quad [x, y, z] correlation length = [3, 3, 20]','fontsize',FS+2,'interpreter','latex')
% title('c) \quad\quad [x, y, z] correlation length = [20, 20, 3]','fontsize',FS+2,'interpreter','latex')

% text(150,315,0,'ly','fontsize',FS,'fontname','Courier') %xlabel('lx','fontsize',FS)
% text(350,160,0,'lx','fontsize',FS,'fontname','Courier') %ylabel('ly','fontsize',FS)
zlabel('\bf{lz}','fontsize',FS)
% title({'Stokes 3D',' '})
text(250,0,280,'b)','fontsize',FS+2,'fontname','Courier')

cb = colorbar;
% cb.Location = 'southoutside';
cb.Limits = [-0.62 0.62];
% cb.Ticks = [0.05 0.95];
% cb.TickLabels = {'0' '1'};
posCB = get(cb,'position');
cb.Position = [posCB(1)*0.9999 posCB(2)*0.99 posCB(3)*0.999 posCB(4)];
% set(cb,'fontsize',FS)
set(gca,'fontsize',FS,'fontname','Courier')
% text(340,180,-100,'A_{initial}','fontsize',FS,'fontname','Courier')

box on
axis image
view(136,18)
camlight
% light
camproj perspective
light('position',[0.6 -1 1]);
% light('position',[1 -1 1]);
light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);

sp5 = subplot(324);
nx = size(Pt_3D,1); ny = size(Pt_3D,2); nz = size(Pt_3D,3);
D = Pt_3D;

s1 = fix( nx/2.0);
s2 = fix( ny/2.0);
s3 = fix( nz/3  );
s4 = fix( nz/2  );
s5 = fix( nz    );

hold on
slice(D( :  , :  ,1:s5),[],s2,s3),shading flat
slice(D( :  , :  ,1:s3),ny,[],[]),shading flat
slice(D(1:s2, :  ,1:s5),ny,[],[]),shading flat
slice(D( :  , :  ,1:s3),[],nx,[]),shading flat
slice(D( :  ,1:s1,1:s4),[],nx,[]),shading flat
slice(D(1:s2, :  , :  ),[],[],s5),shading flat
slice(D( :  ,1:s1, :  ),[],[],s4),shading flat
slice(D( :  , :  ,1:s4),s1,[],[]),shading flat
%%%
is1 = isosurface(D, 1.5);
is2 = isosurface(D,-1.5);
his1 = patch(is1); set(his1,'CData', 2,'Facecolor','Flat','Edgecolor','none')
his2 = patch(is2); set(his2,'CData',-2,'Facecolor','Flat','Edgecolor','none')
hold off
set(gca, 'linewidth',1.4)
set(gca, 'Ticklength', [0 0])
set(gca, 'XTick', [])%[10 nx-5], 'XTicklabel', [10 0], 'fontsize',FS)
set(gca, 'YTick', [])%[10 ny-5], 'YTicklabel', [10 0], 'fontsize',FS)
set(gca, 'ZTick', [10 nz-5], 'ZTicklabel', [0 10], 'fontsize',FS)

% xlabel('Lz = 3000','fontsize',FS,'interpreter','latex')

% title('a) \quad\quad [x, y, z] correlation length = [5, 5, 5]','fontsize',FS+2,'interpreter','latex')
% title('b) \quad\quad [x, y, z] correlation length = [3, 3, 20]','fontsize',FS+2,'interpreter','latex')
% title('c) \quad\quad [x, y, z] correlation length = [20, 20, 3]','fontsize',FS+2,'interpreter','latex')

% text(150,315,0,'ly','fontsize',FS,'fontname','Courier') %xlabel('lx','fontsize',FS)
% text(350,160,0,'lx','fontsize',FS,'fontname','Courier') %ylabel('ly','fontsize',FS)
zlabel('\bf{lz}','fontsize',FS)
text(250,0,280,'d)','fontsize',FS+2,'fontname','Courier')

cb = colorbar;
% cb.Location = 'southoutside';
% cb.Limits = [0 1];
% cb.Ticks = [0.05 0.95];
% cb.TickLabels = {'0' '1'};
posCB = get(cb,'position');
cb.Position = [posCB(1)*0.9999 posCB(2)*0.99 posCB(3)*0.999 posCB(4)];
% set(cb,'fontsize',FS)
set(gca,'fontname','Courier')
% text(340,180,-100,'A_{initial}','fontsize',FS,'fontname','Courier')

box on
axis image
view(136,18)
camlight
% light
camproj perspective
light('position',[0.6 -1 1]);
% light('position',[1 -1 1]);
light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);

sp6 = subplot(326);
nx = size(Txz_3D,1); ny = size(Txz_3D,2); nz = size(Txz_3D,3);
D = Txz_3D;

s1 = fix( nx/2.0);
s2 = fix( ny/2.0);
s3 = fix( nz/3  );
s4 = fix( nz/2  );
s5 = fix( nz    );

hold on
slice(D( :  , :  ,1:s5),[],s2,s3),shading flat
slice(D( :  , :  ,1:s3),ny,[],[]),shading flat
slice(D(1:s2, :  ,1:s5),ny,[],[]),shading flat
slice(D( :  , :  ,1:s3),[],nx,[]),shading flat
slice(D( :  ,1:s1,1:s4),[],nx,[]),shading flat
slice(D(1:s2, :  , :  ),[],[],s5),shading flat
slice(D( :  ,1:s1, :  ),[],[],s4),shading flat
slice(D( :  , :  ,1:s4),s1,[],[]),shading flat
%%%
is1 = isosurface(D, 0.3);
is2 = isosurface(D,-0.3);
his1 = patch(is1); set(his1,'CData', 0.4,'Facecolor','Flat','Edgecolor','none')
his2 = patch(is2); set(his2,'CData',-0.4,'Facecolor','Flat','Edgecolor','none')
hold off
set(gca, 'linewidth',1.4)
set(gca, 'Ticklength', [0 0])
set(gca, 'XTick', [10 nx-5], 'XTicklabel', [10 0], 'fontsize',FS)
set(gca, 'YTick', [10 ny-5], 'YTicklabel', [10 0], 'fontsize',FS)
set(gca, 'ZTick', [10 nz-5], 'ZTicklabel', [0 10], 'fontsize',FS)

% xlabel('Lz = 3000','fontsize',FS,'interpreter','latex')

% title('a) \quad\quad [x, y, z] correlation length = [5, 5, 5]','fontsize',FS+2,'interpreter','latex')
% title('b) \quad\quad [x, y, z] correlation length = [3, 3, 20]','fontsize',FS+2,'interpreter','latex')
% title('c) \quad\quad [x, y, z] correlation length = [20, 20, 3]','fontsize',FS+2,'interpreter','latex')

text(150,315,0,'\bf{ly}','fontsize',FS,'fontname','Courier') %xlabel('lx','fontsize',FS)
text(350,160,0,'\bf{lx}','fontsize',FS,'fontname','Courier') %ylabel('ly','fontsize',FS)
zlabel('\bf{lz}','fontsize',FS)
text(250,0,280,'f)','fontsize',FS+2,'fontname','Courier')

cb = colorbar;
% cb.Location = 'southoutside';
% cb.Limits = [0 1];
% cb.Ticks = [0.05 0.95];
% cb.TickLabels = {'0' '1'};
posCB = get(cb,'position');
cb.Position = [posCB(1)*0.9999 posCB(2)*0.99 posCB(3)*0.999 posCB(4)];
% set(cb,'fontsize',FS)
set(gca,'fontname','Courier')
% text(340,180,-100,'A_{initial}','fontsize',FS,'fontname','Courier')

box on
axis image
view(136,18)
camlight
% light
camproj perspective
light('position',[0.6 -1 1]);
% light('position',[1 -1 1]);
light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);

pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*0.96  pos1(2)*0.96 pos1(3)*1.1 pos1(4)*1.1])
pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)*0.96  pos2(2)*0.94 pos2(3)*1.1 pos2(4)*1.1])
pos3 = get(sp3,'position'); set(sp3,'position',[pos3(1)*0.76   pos3(2)*0.82 pos3(3)*1.1 pos3(4)*1.1])
pos4 = get(sp4,'position'); set(sp4,'position',[pos4(1)*0.86  pos4(2)*0.92 pos4(3)*1.3 pos4(4)*1.3])
pos5 = get(sp5,'position'); set(sp5,'position',[pos5(1)*0.86  pos5(2)*0.88 pos5(3)*1.3 pos5(4)*1.3])
pos6 = get(sp6,'position'); set(sp6,'position',[pos6(1)*0.86  pos6(2)*0.6  pos6(3)*1.3 pos6(4)*1.3])

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_stokes_ve','-dpng','-r300')
end