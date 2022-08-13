clear
close all
data = load('train.mat');

n_trajectories = data.n_trajectories;
%n_trajectories = 64;
Nt = 20;
make_movie = 0;

%plotarg = 'V';
%plotarg = 'dVdX1';
%plotarg = 'dVdX2';
plotarg = 'U';

view_angle = nan;
view_angle = [155,15];
fig_size = [300,275];

labels = ["$x_1$","$x_2$"];%["$x$","$\dot x$"];

Xmin = min(data.X,[],2);
Xmax = max(data.X,[],2);

k0 = [find(data.t == 0), length(data.t)+1];

fig = figure;
fig.Position(3:4) = fig_size;
hold on
box on
axis tight

xlabel(labels(1), 'Interpreter', 'Latex', 'Fontsize', 16)
ylabel(labels(2), 'Interpreter', 'Latex', 'Fontsize', 16)

if strcmp(plotarg, 'V')
    Z = data.V;
    ZLim = [0, max(Z)];
    zlabel('$V (\mathbf x)$', 'Interpreter', 'latex', 'Fontsize', 16)
elseif strcmp(plotarg, 'dVdX1')
    Z = squeeze(data.dVdX(1,:,:));
    ZLim = [-max(Z), max(Z)];%min(Z), max(Z)];
    zlabel('$\partial V / \partial x_1 (\mathbf x)$', 'Interpreter', 'latex', 'Fontsize', 16)
elseif strcmp(plotarg, 'dVdX2')
    Z = squeeze(data.dVdX(2,:,:));
    ZLim = [min(Z), -min(Z)];%max(Z)];
    zlabel('$\partial V / \partial x_2 (\mathbf x)$', 'Interpreter', 'latex', 'Fontsize', 16)
elseif strcmp(plotarg, 'U')
    Z = data.U;
    ZLim = [min(Z), max(Z)];
    zlabel('$u (\mathbf x)$', 'Interpreter', 'latex', 'Fontsize', 16)
end

ax = gca;
ax.FontSize = 12;
ax.TickLabelInterpreter = 'latex';
ax.XLim = [Xmin(1), Xmax(1)];
ax.YLim = [Xmin(2), Xmax(2)];
ax.ZLim = ZLim;

ax.XTickLabel = [];
ax.YTickLabel = [];
ax.ZTickLabel = [];

caxis(ZLim);

if ~all(isnan(view_angle))
    view(view_angle(1),view_angle(2))
end

if make_movie
    fig.Color = 'w';
    clear F

    ax.XTickLabel = [];
    ax.YTickLabel = [];
    ax.ZTickLabel = [];

    if make_movie == 1
        F(k0(n_trajectories+1)) = struct('cdata',[],'colormap',[]);
    elseif make_movie == 2
        F(n_trajectories*(Nt+1)+1) = struct('cdata',[],'colormap',[]);
    elseif make_movie == 3
        F(n_trajectories+1) = struct('cdata',[],'colormap',[]);
    end
    
    j = 1;
    F(j) = getframe(fig);
end

for i=1:n_trajectories
    if make_movie
        if make_movie == 1
            for k=k0(i):k0(i+1)-1
                scatter3(...
                    data.X(1,k),...
                    data.X(2,k),...
                    Z(1,k),...
                    25,...
                    Z(1,k),...
                    '.'...
                )
                drawnow
                j = j + 1;
                F(j) = getframe(fig);
            end
        elseif make_movie == 2
            for k=k0(i):k0(i)+Nt
                scatter3(...
                    data.X(1,k),...
                    data.X(2,k),...
                    Z(1,k),...
                    25,...
                    Z(1,k),...
                    '.'...
                )
                drawnow
                j = j + 1;
                F(j) = getframe(fig);
            end

            scatter3(...
                data.X(1,k0(i)+Nt+1:k0(i+1)-1),...
                data.X(2,k0(i)+Nt+1:k0(i+1)-1),...
                Z(1,k0(i)+Nt+1:k0(i+1)-1),...
                25,...
                Z(1,k0(i)+Nt+1:k0(i+1)-1),...
                '.'...
            )

            drawnow
            j = j + 1;
            F(j) = getframe(fig);
        elseif make_movie == 3
            scatter3(...
                data.X(1,k0(i):k0(i+1)-1),...
                data.X(2,k0(i):k0(i+1)-1),...
                Z(1,k0(i):k0(i+1)-1),...
                25,...
                Z(1,k0(i):k0(i+1)-1),...
                '.'...
            )

            drawnow
            j = j + 1;
            F(j) = getframe(fig);
        end
    else
        scatter3(...
            data.X(1,k0(i):k0(i+1)-1),...
            data.X(2,k0(i):k0(i+1)-1),...
            Z(1,k0(i):k0(i+1)-1),...
            16,...
            Z(1,k0(i):k0(i+1)-1),...
            '.'...
        )
    end
end