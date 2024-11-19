function plotsolution( grid, u )
% This function plots the temperature distribution in the grid...
% Grid: Is one of the grids
% u: Is the solution vector...


clf;
axis off;
hold on;

caxis([min(u) max(u)]);
for i=1:5
  for j=grid.theta{i}'
    fill(grid.coor(j,1),grid.coor(j,2),u(j));
  end
end

title('Temperature distribution')
colorbar;

shading interp;

hold off;

