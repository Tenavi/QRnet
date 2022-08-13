clear
%close all

% Dimension
D = 64;

W = 0;
architecture = 'LQR';
X_NN = nan;
X_opt = nan;
X_LQR = nan;

load(['D', int2str(D), '/results/sim_data.mat'])
params = load(['D', int2str(D), '/params.mat']);

%X = X_LQR; t = t_LQR;
X = X_NN; t = t_NN;
%X = X_opt; t = t_opt;

NN_name = architecture;
NN_name_info = load('../plotting/NN_names.mat');
NN_name_idx = strcmp(NN_name, NN_name_info.NN_names);
NN_legend_name = NN_name_info.legend_names(NN_name_idx);

plot_bvp = 1;
plot_lqr = 0;
plot_nn = 1;

if isnan(X_opt)
    plot_bvp = 0;
    X = X_NN; t = t_NN;
end
if isnan(X_NN)
    plot_nn = 0;
    X = X_LQR; t = t_LQR;
end
if isnan(X_LQR)
    plot_lqr = 0;
    X = X_NN; t = t_NN;
end

T = min([max(t_opt), max(t_LQR), max(t_NN)]);
%T = 90;
T = 15;

%CLIM = 0;
CLIM = [-1,1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xi = [1,reshape(params.xi, [1, D]),-1];
X = [zeros(1,size(X,2)); X; zeros(1,size(X,2))];

fig = figure;
fig.Position(3:4) = [600, 350];

for i = 1:2
    subplot(2,1,i);

    hold on
    box on

    if i == 1
        ax1 = gca;
        ax1.FontSize = 12;
        ax1.TickLabelInterpreter = 'latex';
        ax1.XLim = [0, T];
        ax1.YLim = [-1, 1];

        surf(t',xi',X,'edgecolor','none','facecolor','interp')

        for d=[1,length(xi)]
            plot3(t,xi(d)*ones(size(t)),X(d,:),'k')
        end
        plot3(t(1)*ones(size(xi)),xi,X(:,1),'k')
        plot3(T*ones(size(xi)),xi,X(:,max(find(t <= T))),'k')

        colormap('hot')
        if any(CLIM)
            caxis(CLIM)
        end
        cbar = colorbar;
        cbar.Location = 'southoutside';
        cbar.Orientation = 'horizontal';
        cbar.Position([2,4]) = [0.45,.03];

        title(ax1, '\textbf{Closed-loop dynamics}','FontSize',16,'interpreter','latex')
        ylabel('$\xi$', 'FontSize',16,'interpreter','latex')
        zlabel('$X(t,\xi)$', 'FontSize',16,'interpreter','latex')
    elseif i == 2
        ax2 = gca;
        ax2.FontSize = 12;
        ax2.TickLabelInterpreter = 'latex';
        ax2.XLim = [0, T];
        ax2.YLim = [0, 1.2];

        if plot_bvp
            plot(t_opt,sqrt(params.w' * (X_opt.^2)),'k','DisplayName', 'optimal', 'linewidth', 1.25)
        end
        if plot_lqr
            plot(t_LQR,sqrt(params.w' * (X_LQR.^2)),':','DisplayName', 'LQR',...
                'color', ax2.ColorOrder(1,:),'linewidth', 2)
        end
        if plot_nn
            plot(t_NN,sqrt(params.w' * (X_NN.^2)),'--','DisplayName', NN_legend_name,...
                'color', ax2.ColorOrder(2,:),'linewidth', 2)
        end

        ax2.Position([2,4]) = [.15,.175];

        ylabel('$\Vert X \Vert_{L^2_{(-1,1)}}$', 'FontSize',16,'interpreter','latex')

        lgd = legend;
        lgd.FontSize = 14;
        lgd.Interpreter = 'latex';
        lgd.Location = 'north';
        lgd.Orientation = 'horizontal';
    end

    xlabel('$t$', 'FontSize',16,'interpreter','latex')
end

fig = figure;
fig.Position(3:4) = [600, 275];

for i = 1:2
    subplot(2,1,i);

    if i == 1
        ax3 = gca;
        ax3.FontSize = 12;
        ax3.TickLabelInterpreter = 'latex';
        ax3.XLim = [0, T];

        %ax3.YLim = [-0.7,0.1];
        %ax3.YTick = [-0.6, -0.3, 0];
    elseif i == 2
        ax4 = gca;
        ax4.FontSize = 12;
        ax4.TickLabelInterpreter = 'latex';
        ax4.XLim = [0, T];

        %ax4.YLim = [-0.1,0.7];
        %ax4.YTick = [0, 0.3, 0.6];
    end

    hold on
    box on

    if plot_bvp
        plot(t_opt,U_opt(i,:),'k','DisplayName', 'optimal', 'linewidth', 1.25)
    end
    if plot_lqr
        plot(t_LQR,U_LQR(i,:),':','DisplayName', 'LQR',...
                'color', ax3.ColorOrder(1,:),'linewidth', 2)
    end
    if plot_nn
        plot(t_NN,U_NN(i,:),'--','DisplayName', NN_legend_name,...
                'color', ax3.ColorOrder(2,:),'linewidth', 2)
    end

    xlabel('$t$','FontSize',16,'interpreter','latex')
    if i == 1
        ylab1 = ylabel('$u_1$','FontSize',16,'interpreter','latex');
        title('\textbf{Controls}','FontSize',16,'interpreter','latex')

        lgd = legend;
        lgd.FontSize = 14;
        lgd.Interpreter = 'latex';
        lgd.Location = 'north';
        lgd.Orientation = 'horizontal';
    elseif i == 2
        ylab2 = ylabel('$u_2$','FontSize',16,'interpreter','latex');
    end
end

% Sets the heights of the two plots equal
% ax3.Position(4) = .225;
% ax3.Position(2) = ax3.Position(2) + .075;
% ax4.Position(4) = .225;
% ax4.Position(2) = ax4.Position(2) + .1;

% Aligns the y-labels
ylabpos = min([ylab1.Position(1), ylab2.Position(1)]);
ylab1.Position(1) = ylabpos;
ylab2.Position(1) = ylabpos;
